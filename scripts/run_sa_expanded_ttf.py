"""New-map four-arm TTF confirmation; frozen models, serial solve, deferred audit."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_completion_independent_ttf as old
from scripts import run_sa_completion_amended_ttf as previous
from experiments import sa_expanded_timing as rt
from experiments import sa_completion_endpoint_repair as repair
from experiments.sa_raw_selection_fast import _bind

run, first, require = old.run, old.first, old.require
CONFIG = 'configs/sa_expanded_ttf.json'
CODE = (CONFIG, 'scripts/run_sa_expanded_ttf.py', 'experiments/sa_expanded_timing.py',
        'tests/evaluation/test_sa_expanded_timing.py', 'docs/SA_EXPANDED_TTF_PROTOCOL_ZH.md')
SEEDS = (281, 283, 293)


def config():
    c = run.read_json(ROOT/CONFIG)
    fixed = dict(schema='lns2.sa_expanded_ttf.config.v1', output='build/sa-expanded-four-arm-ttf-v1',
        source='build/sa-completion-independent-ttf-v2', maps=12, densities=[.2, .25], bottleneck_ratio=.4,
        solver_seeds=list(SEEDS), replicas=[0], master_seed=2026092804, stream_seed=2026092805,
        phase='expanded-four-arm-20260928', comparison_arms=list(rt.ARMS), workers=1, audit_workers=20,
        budget_seconds=120., process_fuse_seconds=240., preflight_fuse_seconds=300., audit_fuse_seconds=900.,
        max_decisions=None, execution_node_budget=None, bootstrap=5000, bootstrap_seed=2026092806,
        timing_mode=rt.TIMING_MODE, no_training=True, automatic_promotion=False, replacement_permitted=False,
        generation_policy='original_sampler_then_d25_capacity_matching_v1')
    require(all(c[k] == v for k, v in fixed.items()), 'frozen expanded scope changed')
    return c


def prepare():
    from scripts.run_sa_independent_confirmation import historical_inventory
    c = config()
    out = ROOT/c['output']
    require(not out.exists(), 'already registered; inspect/resume')
    source, src = previous.verify()
    require(run.sha256_file(src/'report.json') == c['source_report_sha256'], 'source report changed')
    history = historical_inventory(exclude=out)
    # Recent case registries include matched tasks absent from dataset manifests.
    for p in sorted((ROOT/'build').glob('*/cases.json')):
        doc = run.read_json(p)
        if isinstance(doc, dict) and 'map_seeds' in doc and 'task_seeds' in doc:
            history['seeds'] = sorted(set(history['seeds']) | set(doc['map_seeds']) | set(doc['task_seeds']))
            history['map_hashes'] = sorted(set(history['map_hashes']) | set(doc.get('map_sha256', {}).values()))
            history['manifests'][p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
    rng = random.Random(c['master_seed'])
    masters = [rng.randrange(1, 2**31) for _ in range(c['maps'])]
    require(len(set(masters)) == 12 and not set(masters)&set(history['seeds']), 'master overlap; no redraw')
    models = {}
    for arm in ('parent', 'completion'):
        j = next(j for j in source['jobs'] if j['comparison_arm'] == arm)
        require(j['model']['sha256'] == c[arm+'_sha256'], 'model changed')
        models[arm] = dict(model=j['model'], iteration=j['iteration'])
    plan = deepcopy(source['jobs'][0]['plan'])
    plan['config'].update(output=c['output'], stream_seed=c['stream_seed'])
    require(plan['proposal']['max_decisions'] is None, 'inherited decision cap')
    inputs = source['inputs'] | history['manifests']
    for path in list(CODE) + [str((src/n).relative_to(ROOT).as_posix()) for n in ('registration.json', 'report.json')]:
        inputs[path] = run.sha256_file(run.contained_file(ROOT, path, field='expanded input'))
    body = dict(schema='lns2.sa_expanded_ttf.design.v1', config=c, inputs=inputs, models=models,
        source_plan=plan, history=history, map_masters=masters,
        source_commit=run.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    body['binding'] = run.json_fingerprint(body)
    with first.recovery.strict_lock(out, body['binding'], 'expanded-prepare'):
        run.once(out/'design.json', run.sealed(body))
    return dict(maps=12, tasks=24, conditions=72, episodes=288, timing_started=False,
                planning_budget_upper_hours=9.6, process_fuse_upper_hours=19.2)


verify_design = _bind(old.verify_design, config=config)
check_isolation = old.check_isolation


def generate_task(md, tc, seed, task_id, template):
    from generators.task_flows import generate_tasks
    try:
        return generate_tasks(md, tc, seed, task_id), False, None
    except ValueError as error:
        if tc['agent_density'] != .25 or template is None or not str(error).startswith('unable to sample valid endpoints for agent '):
            raise
        return repair.repair_task(md, tc, seed, task_id, template), True, str(error)


def generate_worker(job):
    from generators.config import merge_dicts
    from generators.io import write_map_bundle, write_instance_bundle
    from generators.validation import validate_map, validate_task
    from generators.warehouse import generate_warehouse
    r, index = job['registration'], job['index']
    c = r['config']
    folder = ROOT/c['output']/f'dataset/shards/{index:02d}'
    require(not folder.exists(), 'partial generation; inspect, no redraw')
    base = run.read_json(ROOT/c['dataset_base'])
    rng = random.Random(r['map_masters'][index])
    md = generate_warehouse(base['map'], rng.randrange(1, 2**31),
                            f'sa_expanded_v1_m{index:02d}_station_centric_0000')
    validate_map(md)
    write_map_bundle(folder/'maps', md)
    rows, template = [], None
    for ti, density in enumerate(c['densities']):
        seed = rng.randrange(1, 2**31)
        task_id = f'{md.map_id}__task_{ti:04d}'
        tc = merge_dicts(base['task'], base['task_variants'][0]['task'])
        tc.update(agent_density=density, required_bottleneck_crossing_ratio=c['bottleneck_ratio'])
        task, amended, reason = generate_task(md, tc, seed, task_id, template)
        validate_task(md, task)
        repair.validate_constraints(md, task, tc)
        write_instance_bundle(folder/'instances', md, task)
        template = task.metadata
        rows.append(dict(map_id=md.map_id, map_seed=md.seed, task_id=task_id, task_seed=seed,
            map_file=(folder/'maps'/(md.map_id+'.map')).relative_to(ROOT).as_posix(),
            scenario_file=(folder/'instances'/(task_id+'.scen')).relative_to(ROOT).as_posix(),
            task_file=(folder/'instances'/(task_id+'.json')).relative_to(ROOT).as_posix(),
            task_variant=f'bottleneck_d{int(100*density)}', agent_count=task.agent_count,
            endpoint_amended=amended, sampling_failure=reason))
    receipt = run.sealed(dict(binding=r['binding'], rows=rows,
        files={p.relative_to(ROOT).as_posix():run.sha256_file(p) for p in sorted(folder.rglob('*')) if p.is_file()}))
    run.once(folder/'receipt.json', receipt)
    return receipt


def generate():
    from lns2_selector.evaluation.path_quality_preflight import audit_task
    r, out = verify_design()
    with first.recovery.strict_lock(out, r['binding'], 'expanded-generate'):
        require(not (out/'cases.json').exists(), 'already generated')
        shards, pending = [], []
        for i in range(12):
            p = out/f'dataset/shards/{i:02d}/receipt.json'
            if p.exists():
                shard = run.check_seal(run.read_json(p))
                require(shard['binding'] == r['binding'], 'stale shard')
                for name, sha in shard['files'].items():
                    require(run.sha256_file(ROOT/name) == sha, 'changed generated file: '+name)
                shards.append(shard)
            else:
                pending.append(dict(registration=r, index=i))
        if pending:
            with ProcessPoolExecutor(max_workers=min(20, len(pending))) as pool:
                shards += list(pool.map(generate_worker, pending))
        rows = sorted([row for shard in shards for row in shard['rows']], key=lambda x:x['task_id'])
        maps = check_isolation(rows, r['history'])
        cases = []
        for row in rows:
            checked = audit_task(ROOT/row['map_file'], ROOT/row['scenario_file'], ROOT/row['task_file'], row['agent_count'])
            cases.append(dict(task_id=row['task_id'], map_id=row['map_id'], family='warehouse',
                solver_seeds=list(SEEDS), static_audit=checked, status='static_ready_runtime_unverified',
                task_variant=row['task_variant'], density=.2 if row['task_variant']=='bottleneck_d20' else .25,
                endpoint_amended=row['endpoint_amended'],
                files={k:row[k] for k in ('map_file', 'scenario_file', 'task_file')}))
        run.once(out/'cases.json', run.sealed(dict(binding=r['binding'], cases=cases, map_sha256=maps,
            files={k:v for shard in shards for k,v in shard['files'].items()},
            map_seeds=sorted({row['map_seed'] for row in rows}), task_seeds=sorted({row['task_seed'] for row in rows}))))
    return dict(maps=12, tasks=24, amended_tasks=sum(c['endpoint_amended'] for c in cases), no_solver_calls=True)


condition_jobs = old.condition_jobs
qualify = _bind(old.qualify, verify_design=verify_design, rt=rt)


def schedule(jobs):
    jobs = list(jobs)
    groups = {}
    for j in jobs:
        group = groups.setdefault((j['pair_id'], j['replica']), {})
        require(j['comparison_arm'] not in group, 'duplicate arm')
        group[j['comparison_arm']] = j
    require(len(groups) == 72, 'expected 72 conditions')
    maps = sorted({j['case']['map_id'] for j in jobs})
    require(len(maps) == 12, 'expected 12 maps')
    orders = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
    result = []
    for _, group in sorted(groups.items()):
        require(set(group) == set(rt.ARMS), 'missing arm')
        anchor = group['parent']
        for field in ('expected_initial', 'case', 'solver_seed', 'phase', 'replica'):
            require(all(group[a][field] == anchor[field] for a in rt.ARMS), 'unpaired '+field)
        require(len({group[a]['plan']['config']['stream_seed'] for a in rt.ARMS}) == 1, 'unpaired stream')
        variant = anchor['case']['task_variant']
        require(variant in ('bottleneck_d20', 'bottleneck_d25') and anchor['solver_seed'] in SEEDS and anchor['replica'] == 0,
                'unexpected condition')
        offset = 3*int(variant == 'bottleneck_d25') + SEEDS.index(anchor['solver_seed'])
        order = orders[(maps.index(anchor['case']['map_id']) + offset) % 4]
        for index in order:
            result.append(dict(group[rt.ARMS[index]], schedule_index=len(result)))
    for arm in rt.ARMS:
        js = [j for j in result if j['comparison_arm'] == arm]
        require(sorted(Counter(j['case']['map_id'] for j in js).values()) == [6]*12, 'map coverage')
        for m in maps:
            require({(j['case']['task_variant'], j['solver_seed']) for j in js if j['case']['map_id'] == m}
                == {(v,s) for v in ('bottleneck_d20','bottleneck_d25') for s in SEEDS}, 'task/seed coverage')
    return result


def register():
    r, out = verify_design()
    conditions = condition_jobs(r, out)
    first.source.check_complete(r, out, 'qualification', conditions)
    jobs = []
    for j in conditions:
        q = run.check_seal(run.read_json(out/'qualification'/(j['job_id']+'.json')))
        for arm in rt.ARMS:
            spec = r['models'].get(arm, dict(model=None, iteration=0))
            jobs.append(dict(j, **spec, comparison_arm=arm,
                arm=arm if arm in ('official','official_sa') else 'trained_actor',
                job_id=run.json_fingerprint(['expanded-ttf-v1', j['pair_id'], arm])[:24],
                expected_initial=q['initial_fingerprint'], budget_seconds=120., timing_mode=rt.TIMING_MODE,
                max_decisions=None, execution_node_budget=None))
    inputs = r['inputs'] | run.check_seal(run.read_json(out/'cases.json'))['files']
    for p in [out/n for n in ('design.json','cases.json','qualification.json','qualification.complete.json')] + list((out/'qualification').glob('*.json')):
        inputs[p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
    body = dict(schema='lns2.sa_expanded_ttf.registration.v1', config=r['config'], inputs=inputs,
        jobs=schedule(jobs), design_binding=r['binding'], runtime_labels=rt.LABELS,
        source_commit=run.subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip())
    body['binding'] = run.json_fingerprint(body)
    with first.recovery.strict_lock(out, body['binding'], 'expanded-register'):
        run.once(out/'registration.json', run.sealed(body))
    return dict(episodes=288, workers=1, automatic_promotion=False)


verify = _bind(old.verify, config=config, schedule=schedule)
phase = _bind(old.phase, verify=verify, rt=rt)
_collect = _bind(first.collect, verify=verify, rt=rt)
collect = _bind(old.collect, verify=verify, _collect=_collect)
_report = _bind(old.report, verify=verify, rt=rt)


def report():
    r, out = verify()
    if not (out/'report.json').exists():
        _report()
    else:
        for name in ('collect','audit'):
            first.source.check_complete(r,out,name,first.source.jobs_for(r))
    doc = run.check_seal(run.read_json(out/'report.json'))
    require(doc['binding'] == r['binding'], 'stale report')
    amended = {j['case']['task_id'] for j in r['jobs'] if j['case']['endpoint_amended']}
    selected = [row for row in doc['episodes'] if row['task_id'] not in amended]
    path = out/'generation_sensitivity.json'
    if not path.exists():
        run.once(path, run.sealed(dict(binding=r['binding'], report_sha256=run.sha256_file(out/'report.json'),
            amended_tasks=sorted(amended), excluding_amended_tasks=rt.summarize(selected) if selected else None)))
    return {k:v for k,v in doc.items() if k not in ('episodes','by_map','by_density','integrity')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare','generate','qualify','register','verify','preflight',
                                    'collect','resume','audit','report','stop','all'))
    args = p.parse_args()
    if args.phase == 'verify':
        r, _ = verify()
        result = dict(inputs=len(r['inputs']), episodes=len(r['jobs']))
    elif args.phase == 'stop':
        out = ROOT/config()['output']
        require(out.exists(), 'unregistered')
        run.write_json(out/'STOP_AFTER_EPISODE', dict(requested=True))
        result = dict(stop_after_current_episode=True)
    elif args.phase in ('preflight','audit'):
        result = phase(args.phase)
    elif args.phase == 'resume':
        result = collect(True)
    elif args.phase == 'all':
        result = collect()
        if not result.get('paused'):
            phase('audit')
            result = report()
    else:
        result = globals()[args.phase]()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
