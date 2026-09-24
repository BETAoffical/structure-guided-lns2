"""Freeze raw-1 and compare four SA selectors on new warehouse geometry."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_onpolicy as run
from scripts import run_sa_raw_residual as prior
from scripts import collect_sa_second_batch as batch
from scripts import recover_sa_onpolicy as recovery
from experiments import sa_uncapped_runtime as runtime
from experiments.sa_raw_residual_runtime import actor_scope
from experiments.sa_raw_confirmation import ARMS, schedule, summarize

CONFIG = 'configs/sa_raw_independent_confirmation.json'
require = run.require
CODE = (CONFIG, 'scripts/confirm_sa_raw_independent.py', 'experiments/sa_raw_confirmation.py',
        'tests/evaluation/test_sa_raw_confirmation.py', 'docs/SA_RAW_INDEPENDENT_PROTOCOL_ZH.md')


def config():
    c = run.read_json(ROOT/CONFIG)
    require(c['arms'] == list(ARMS) and c['maps'] == 6 and c['replicas'] == 2 and
            c['solver_seeds'] == [251,257] and c['densities'] == [.2,.25] and c['workers'] == 20 and
            c['max_decisions'] is None and c['no_training'] and not c['formal_ttf'] and
            not c['automatic_promotion'], 'frozen scope')
    return c


def prepare():
    from scripts.run_sa_independent_confirmation import historical_inventory
    c = config()
    out = ROOT/c['output']
    require(not out.exists(), 'existing registration; inspect/resume')
    old, source = prior.verify()
    require(source == ROOT/c['source'], 'source output')
    require(run.sha256_file(source/'report.json') ==
            'cdab0622d9bd9b36f1d5409cf108378f367b599330c71315371204cdd19a2fc3', 'source report')
    models = {a: prior.model_spec(old, i) for a,i in [('raw_parent',0),('raw_updated',1)]}
    require(models['raw_updated']['sha256'] ==
            'fad8c8c64d68aefbd19bcfa620eb36f479fbd95d835d819ca9fd30ae2021fe84', 'updated frozen model')
    history = historical_inventory(exclude=out)
    rng = random.Random(c['master_seed'])
    masters = [rng.randrange(1,2**31) for _ in range(c['maps'])]
    require(len(set(masters)) == 6 and not set(masters)&set(history['seeds']), 'master overlap; no redraw')
    inputs = dict(old['inputs'])
    inputs.update(history['manifests'])
    paths = [ROOT/name for name in CODE]+[ROOT/c['dataset_base'],source/'execution_registration.json',
        source/'report.json',source/'comparison.audit.json',source/'update.json',
        source/'parity.windows.json',source/'parity.wsl.json']
    paths += [ROOT/s['path'] for s in models.values()]
    paths += list((ROOT/'generators').rglob('*.py'))
    for path in paths:
        inputs[path.relative_to(ROOT).as_posix()] = run.sha256_file(path)
    body = dict(config=c, models=models, inputs=inputs, source_plan=old['source_plan'],
        history=history, map_masters=masters, no_training=True, no_ttf=True,
        source_commit=run.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    body['binding'] = run.json_fingerprint(body)
    with recovery.strict_lock(out,body['binding'],'confirmation-prepare'):
        run.once(out/'registration.json',run.sealed(body))
    return dict(maps=6,tasks=12,conditions=24,episodes=192,workers=20,no_training=True,no_ttf=True,
                historical_missing_maps=len(history['missing_map_files']))


def verify():
    c = config()
    out = ROOT/c['output']
    reg = run.check_seal(run.read_json(out/'registration.json'))
    require(reg['config'] == c and reg['binding'] == run.json_fingerprint(
        {k:v for k,v in reg.items() if k not in ('binding','integrity')}), 'registration changed')
    for name,sha in reg['inputs'].items():
        require(run.sha256_file(run.contained_file(ROOT,name,field='confirmation input')) == sha, 'changed input: '+name)
    return reg,out


def generate_worker(job):
    from generators.dataset import generate_dataset
    from experiments._common import read_jsonl
    reg,index = job['reg'],job['index']
    c = reg['config']
    folder = ROOT/c['output']/f'dataset/shards/{index:02d}'
    if (folder/'receipt.json').exists():
        receipt = run.check_seal(run.read_json(folder/'receipt.json'))
        require(receipt['binding'] == reg['binding'], 'shard binding')
        for name,sha in receipt['files'].items():
            require(run.sha256_file(ROOT/name) == sha, 'shard changed')
        return receipt
    require(not folder.exists(), 'partial generation; do not redraw')
    base = run.read_json(ROOT/c['dataset_base'])
    base.update(master_seed=reg['map_masters'][index], map_id_prefix=f'sa_raw_confirm_v1_m{index:02d}',
                tasks_per_map=2, splits={'confirmation':{'layout_counts':{'station_centric':1}}})
    template = deepcopy(base['task_variants'][0])
    base['task_variants'] = []
    for density in c['densities']:
        v = deepcopy(template)
        v['name'] = f'bottleneck_d{int(density*100)}'
        v['task'].update(agent_density=density,required_bottleneck_crossing_ratio=c['bottleneck_ratio'])
        base['task_variants'].append(v)
    generate_dataset(base,output_override=folder.relative_to(ROOT))
    rows = read_jsonl(folder/'confirmation/manifest.jsonl')
    for row in rows:
        for key in ('map_file','map_metadata_file','task_file','scenario_file','instance_file','legacy_instance_file'):
            row[key] = (folder/'confirmation'/row[key]).relative_to(ROOT).as_posix()
    receipt = run.sealed(dict(binding=reg['binding'], rows=rows,
        files={p.relative_to(ROOT).as_posix():run.sha256_file(p) for p in sorted(folder.rglob('*')) if p.is_file()}))
    run.once(folder/'receipt.json',receipt)
    return receipt


def generate():
    from lns2_selector.evaluation.path_quality_preflight import audit_task
    reg,out = verify()
    with recovery.strict_lock(out,reg['binding'],'confirmation-generate'):
        with ProcessPoolExecutor(max_workers=6) as pool:
            shards = list(pool.map(generate_worker,[dict(reg=reg,index=i) for i in range(6)]))
        rows = [r for s in shards for r in s['rows']]
        seeds = {r['map_seed'] for r in rows}|{r['task_seed'] for r in rows}
        hashes = {r['map_id']:run.sha256_file(ROOT/r['map_file']) for r in rows}
        require(len(rows) == 12 and len(seeds) == 18 and len(set(hashes.values())) == 6, 'unique task/map coverage')
        require(not seeds&set(reg['history']['seeds']) and not set(hashes.values())&set(reg['history']['map_hashes']),
                'historical seed/geometry overlap; no replacement')
        cases = []
        for r in rows:
            checked = audit_task(ROOT/r['map_file'],ROOT/r['scenario_file'],ROOT/r['task_file'],r['agent_count'])
            cases.append(dict(task_id=r['task_id'],map_id=r['map_id'],family='warehouse',
                solver_seeds=reg['config']['solver_seeds'],static_audit=checked,
                status='static_ready_runtime_unverified',task_variant=r['task_variant'],
                density=r.get('agent_density'),files={k:r[k] for k in ('map_file','scenario_file','task_file')}))
        run.once(out/'cases.json',run.sealed(dict(binding=reg['binding'],cases=cases,
            files={k:v for s in shards for k,v in s['files'].items()},map_sha256=hashes,
            map_seeds=sorted({r['map_seed'] for r in rows}),task_seeds=sorted({r['task_seed'] for r in rows}))))
    return dict(maps=6,tasks=12,seed_overlap=0,geometry_overlap=0,no_training=True)


def conditions(reg,out):
    receipt = run.check_seal(run.read_json(out/'cases.json'))
    require(receipt['binding'] == reg['binding'], 'case binding')
    for name,sha in receipt['files'].items():
        require(run.sha256_file(ROOT/name) == sha, 'case changed')
    return [dict(case=case,solver_seed=seed,split='independent_confirmation',pair_id=f"{case['task_id']}-s{seed}")
            for case in receipt['cases'] for seed in case['solver_seeds']]


def runtime_plan(reg,lane):
    p = deepcopy(reg['source_plan'])
    c = reg['config']
    p['config']['output'] = c['output']+'/'+lane
    p['proposal'].update({k:c[k] for k in ('max_decisions','decision_feature_reference','node_budget',
        'pp_safety_seconds','episode_safety_seconds','process_fuse_seconds')})
    return p


def qjobs(reg,out):
    return [dict(c,phase='qualify-raw-independent',plan=runtime_plan(reg,'qualification'),
        job_id=run.json_fingerprint(['qualification',c['pair_id']])[:24],parent_pid=os.getpid(),
        collection_binding=reg['binding']) for c in conditions(reg,out)]


def qread(reg,j):
    row = run.result_read(batch.qfolder(j),j['plan'])
    require(row['execution_binding'] == reg['binding'] and row['status'] == 'ok' and
        all(row[k] == j[k] for k in ('job_id','pair_id','split')) and row['map_id'] == j['case']['map_id'], 'qualification identity')
    state = run.read_json(batch.qfolder(j)/'initial.json')
    from scripts.run_sa_path_quality import state_fingerprint
    require(state_fingerprint(state) == row['initial_fingerprint'] and
        row['initial_conflicts'] == state['num_of_colliding_pairs'] and row['feasible'] == state['feasible'] and
        row['paths_signature'] == batch.paths_signature(state,row['map_id']), 'qualification state')
    return row


def qualify(resume=False):
    reg,out = verify()
    jobs = qjobs(reg,out)
    with recovery.strict_lock(out,reg['binding'],'confirmation-qualify'):
        done = batch.execute_batches(reg,out,jobs,batch.qualification_worker,batch.qfolder,lambda j:qread(reg,j),
            'qualification',resume,reg['config']['qualification_fuse_seconds'])
        if not done:return dict(status='paused_or_needs_inspection')
        rows = [qread(reg,j) for j in jobs]
        nonzero = [r for r in rows if not r['feasible']]
        allowed = (len(nonzero) >= reg['config']['minimum_nonzero_conditions'] and
            len({r['map_id'] for r in nonzero}) >= reg['config']['minimum_active_maps'])
        run.once(out/'qualification.json',run.sealed(dict(binding=reg['binding'],rows=rows,valid=len(rows),
            nonzero=len(nonzero),active_maps=len({r['map_id'] for r in nonzero}),collection_allowed=allowed,
            complete_sha256=run.sha256_file(out/'qualification.complete.json'))))
    return dict(valid=len(rows),nonzero=len(nonzero),collection_allowed=allowed,no_redraw=True)


def ready_jobs(reg,out):
    proof = run.check_seal(run.read_json(out/'qualification.json'))
    complete = run.check_seal(run.read_json(out/'qualification.complete.json'))
    qs = qjobs(reg,out)
    rows = [qread(reg,j) for j in qs]
    require(proof['binding'] == complete['binding'] == reg['binding'] and proof['rows'] == rows and
        proof['complete_sha256'] == run.sha256_file(out/'qualification.complete.json') and
        complete['jobs'] == proof['valid'] == len(rows) == 24 and proof['collection_allowed'], 'qualification gate')
    require(complete['files'] == {j['job_id']:run.sha256_file(batch.qfolder(j)/'result.json') for j in qs}, 'qualification coverage')
    initial = {r['pair_id']:r['initial_fingerprint'] for r in rows}
    return [dict(j,plan=runtime_plan(reg,'runs'),model=reg['models'].get(j['comparison_arm']),
                 parent_pid=os.getpid(),comparison_binding=reg['binding'],expected_initial=initial[j['pair_id']])
            for j in schedule(conditions(reg,out),reg['config']['phase'],reg['config']['replicas'])]


def scope(job):
    if job['comparison_arm'] in ('raw_parent','raw_updated'):
        bundle = prior.load_model(job)
        return actor_scope(bundle,ROOT/job['plan']['config']['output'],job['plan'])
    require(job['arm'] == job['comparison_arm'] and job['arm'] in ('dual16_sa','official_sa') and job['model'] is None,
            'baseline identity')
    return nullcontext()


def worker(job):
    with scope(job):
        return runtime.episode_worker(job)


def audit_worker(job):
    with scope(job):
        return runtime.audit_worker(job)


def read_result(job):
    row = run.result_read(prior.folder(job),job['plan'])
    require(row['execution_binding'] == job['comparison_binding'] and
        all(row[k] == job[k] for k in ('job_id','pair_id','replica','split','arm')) and
        row['map_id'] == job['case']['map_id'] and row['initial_fingerprint'] == job['expected_initial'], 'result pairing')
    sha = job['model']['policy_sha256'] if job['model'] else job['arm']
    require(row['policy_sha256'] == sha, 'result model')
    runtime.validate_terminal(row,job['plan']['proposal'])
    return dict(row,comparison_arm=job['comparison_arm'])


def collect(resume=False):
    reg,out = verify()
    jobs = ready_jobs(reg,out)
    with recovery.strict_lock(out,reg['binding'],'confirmation-collect'):
        done = batch.execute_batches(reg,out,jobs,worker,prior.folder,read_result,'comparison',resume,
                                     reg['config']['process_fuse_seconds'])
    return dict(complete=done,episodes=len(jobs),no_ttf=True)


def audited(reg,out):
    jobs = ready_jobs(reg,out)
    complete = run.check_seal(run.read_json(out/'comparison.complete.json'))
    require(complete['binding'] == reg['binding'] and complete['jobs'] == len(jobs), 'incomplete collection')
    require(complete['files'] == {j['job_id']:run.sha256_file(prior.folder(j)/'result.json') for j in jobs}, 'changed results')
    return jobs,complete


def audit():
    from experiments.repair_collection import _run_jobs
    reg,out = verify()
    jobs,_ = audited(reg,out)
    with recovery.strict_lock(out,reg['binding'],'confirmation-audit'):
        results = _run_jobs(audit_worker,jobs,20,phase='audit',output_root=out/'audit-progress',
            run_fingerprint=reg['binding'],timeout_seconds=960.)
        require(len(results) == len(jobs) and all(r['status'] == 'ok' for r in results), 'full audit failed')
        run.once(out/'comparison.audit.json',run.sealed(dict(binding=reg['binding'],results=results)))
    return dict(audited=len(results))


def report():
    reg,out = verify()
    jobs,complete = audited(reg,out)
    proof = run.check_seal(run.read_json(out/'comparison.audit.json'))
    require(proof['binding'] == reg['binding'] and len(proof['results']) == len(jobs) and
        all(r['status'] == 'ok' for r in proof['results']) and
        {r['job_id']:r['result_sha256'] for r in proof['results']} == complete['files'], 'stale audit')
    rows = [read_result(j) for j in jobs]
    summary = summarize(rows,conditions(reg,out),reg['config']['replicas'],
                        reg['config']['bootstrap'],reg['config']['bootstrap_seed'])
    run.once(out/'report.json',run.sealed(dict(binding=reg['binding'],summary=summary,episodes=rows,
        audit_sha256=run.sha256_file(out/'comparison.audit.json'))))
    run.write_json(out/'run_status.json',dict(status='complete',no_training=True,no_promotion=True,no_ttf=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','verify','generate','qualify','dry-run','collect','audit','report','stop'])
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare':value=prepare()
    elif args.phase == 'verify':value=dict(verified=True,inputs=len(verify()[0]['inputs']))
    elif args.phase == 'generate':value=generate()
    elif args.phase == 'qualify':value=qualify(args.resume)
    elif args.phase == 'collect':value=collect(args.resume)
    elif args.phase == 'audit':value=audit()
    elif args.phase == 'report':value=report()
    elif args.phase == 'dry-run':
        reg,out=verify()
        value=dict(jobs=len(ready_jobs(reg,out)),workers=20,max_decisions=None,
                   maximum_fuse_waves_minutes=10*960/60,no_ttf=True)
    else:
        _,out=verify()
        (out/'STOP_AFTER_BATCH').touch(exist_ok=True)
        value=dict(stop_after_current_batch=True)
    print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)


if __name__ == '__main__':main()
