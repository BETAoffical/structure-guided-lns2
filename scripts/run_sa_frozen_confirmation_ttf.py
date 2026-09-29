"""Independent frozen A/A2 timing: prepare separately, then serial collect and deferred audit."""
import argparse
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
from scripts import run_sa_expanded_ttf as old
from scripts import run_sa_completion_continuation as continuation
from experiments import sa_frozen_confirmation_timing as rt
from experiments.sa_raw_selection_fast import _bind

run, first, require = old.run, old.first, old.require
CONFIG = 'configs/sa_frozen_confirmation_ttf.json'
CODE = (CONFIG, 'scripts/run_sa_frozen_confirmation_ttf.py',
        'experiments/sa_frozen_confirmation_timing.py',
        'tests/evaluation/test_sa_frozen_confirmation_timing.py',
        'docs/SA_FROZEN_CONFIRMATION_TTF_PROTOCOL_ZH.md')
SEEDS = (307, 311, 313)


def config():
    c = run.read_json(ROOT/CONFIG)
    fixed = dict(schema='lns2.sa_frozen_confirmation_ttf.config.v1',
        output='build/sa-frozen-confirmation-ttf-v1', source='build/sa-expanded-four-arm-ttf-v1',
        continuation='build/sa-completion-continuation-v1', maps=12, densities=[.2, .25],
        bottleneck_ratio=.4, solver_seeds=list(SEEDS), replicas=[0],
        master_seed=2026093007, stream_seed=2026093008, bootstrap_seed=2026093009,
        map_id_prefix='sa_frozen_confirm_v1', phase='frozen-A-A2-four-arm-20260930',
        comparison_arms=list(rt.ARMS), workers=1, audit_workers=20, budget_seconds=120.,
        process_fuse_seconds=240., preflight_fuse_seconds=300., audit_fuse_seconds=900.,
        max_decisions=None, execution_node_budget=None, bootstrap=5000,
        timing_mode=rt.TIMING_MODE, no_training=True, automatic_promotion=False,
        replacement_permitted=False, generation_policy='original_sampler_then_d25_capacity_matching_v1',
        primary_contrast='parent_vs_official_sa', exploratory_contrast='completion_vs_parent',
        parent_sha256='81727429f423734aef544c1e12ad61fae965a2118898fadafeab699dd9307062',
        completion_sha256='9ba2192c53f02563c02b8f09637d69a5448ccfb7d847f1f88de4f642784d15a9')
    require(all(c[k] == v for k, v in fixed.items()), 'frozen confirmation scope changed')
    return c


def prepare():
    from scripts.run_sa_independent_confirmation import historical_inventory
    c = config()
    out = ROOT/c['output']
    require(not out.exists(), 'already registered; inspect/resume')
    source, src = old.verify()
    trained, train_out = continuation.verify()
    require(src == ROOT/c['source'] and train_out == ROOT/c['continuation'], 'source directory identity')
    require(run.sha256_file(src/'report.json') == c['source_report_sha256'], 'timing report changed')
    require(run.sha256_file(train_out/'report.json') == c['continuation_report_sha256'], 'continuation report changed')
    models = {}
    for arm, iteration in (('parent', 2), ('completion', 3)):
        spec = continuation.model_spec(trained, arm)
        require(spec['sha256'] == c[arm+'_sha256'], 'frozen model changed')
        bundle = run.read_json(ROOT/spec['path'])
        require(bundle['iteration'] == iteration, 'model iteration mismatch')
        models[arm] = dict(model=spec, iteration=iteration)
    history = historical_inventory(exclude=out)
    for p in sorted((ROOT/'build').glob('*/cases.json')):
        doc = run.read_json(p)
        if isinstance(doc, dict) and 'map_seeds' in doc and 'task_seeds' in doc:
            history['seeds'] = sorted(set(history['seeds']) | set(doc['map_seeds']) | set(doc['task_seeds']))
            history['map_hashes'] = sorted(set(history['map_hashes']) | set(doc.get('map_sha256', {}).values()))
            history['manifests'][p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
    rng = random.Random(c['master_seed'])
    masters = [rng.randrange(1, 2**31) for _ in range(c['maps'])]
    require(len(set(masters)) == 12 and not set(masters)&set(history['seeds']), 'master overlap; no redraw')
    plan = deepcopy(source['jobs'][0]['plan'])
    plan['config'].update(output=c['output'], stream_seed=c['stream_seed'])
    require(plan['proposal']['max_decisions'] is None and
            plan['template']['environment']['max_repair_iterations'] == 0, 'inherited decision cap')
    inputs = source['inputs'] | history['manifests']
    paths = list(CODE) + [models[a]['model']['path'] for a in models]
    paths += [(src/n).relative_to(ROOT).as_posix() for n in ('registration.json', 'report.json')]
    paths += [(train_out/n).relative_to(ROOT).as_posix()
              for n in ('registration.json', 'update.completion.json', 'report.json', 'comparison.audit.json')]
    for path in paths:
        inputs[path] = run.sha256_file(run.contained_file(ROOT, path, field='confirmation input'))
    body = dict(schema='lns2.sa_frozen_confirmation_ttf.design.v1', config=c, inputs=inputs,
        models=models, source_plan=plan, history=history, map_masters=masters, runtime_labels=rt.LABELS,
        source_commit=run.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    body['binding'] = run.json_fingerprint(body)
    with first.recovery.strict_lock(out, body['binding'], 'frozen-confirmation-prepare'):
        run.once(out/'design.json', run.sealed(body))
    return dict(maps=12, tasks=24, conditions=72, episodes=288, timing_started=False,
                planning_budget_upper_hours=9.6, process_fuse_upper_hours=19.2)


verify_design = _bind(old.verify_design, config=config)
check_isolation = old.check_isolation


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
                            f"{c['map_id_prefix']}_m{index:02d}_station_centric_0000")
    validate_map(md)
    write_map_bundle(folder/'maps', md)
    rows, template = [], None
    for ti, density in enumerate(c['densities']):
        seed = rng.randrange(1, 2**31)
        task_id = f'{md.map_id}__task_{ti:04d}'
        tc = merge_dicts(base['task'], base['task_variants'][0]['task'])
        tc.update(agent_density=density, required_bottleneck_crossing_ratio=c['bottleneck_ratio'])
        task, amended, reason = old.generate_task(md, tc, seed, task_id, template)
        validate_task(md, task)
        old.repair.validate_constraints(md, task, tc)
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


generate = _bind(old.generate, verify_design=verify_design, generate_worker=generate_worker, SEEDS=SEEDS)
condition_jobs = old.condition_jobs
qualify = _bind(old.qualify, verify_design=verify_design, rt=rt)
schedule = _bind(old.schedule, SEEDS=SEEDS, rt=rt)
register = _bind(old.register, verify_design=verify_design, schedule=schedule, rt=rt)
verify = _bind(old.verify, config=config, schedule=schedule)
phase = _bind(old.phase, verify=verify, rt=rt)
_collect = _bind(first.collect, verify=verify, rt=rt)
collect = _bind(old.collect, verify=verify, _collect=_collect)
_report = _bind(old._report, verify=verify, rt=rt)
report = _bind(old.report, verify=verify, _report=_report, rt=rt)


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
