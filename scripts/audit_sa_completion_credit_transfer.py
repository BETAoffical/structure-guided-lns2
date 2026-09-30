"""Read two existing on-policy A batches; audit credit transfer without updating any policy."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import sa_completion_credit_transfer as diagnostic
from scripts import run_sa_completion_continuation as continuation
from scripts import run_sa_completion_map_coverage as coverage

run, raw, require = continuation.run, continuation.raw, continuation.require
CONFIG = 'configs/sa_completion_credit_transfer.json'
SOURCES = {'A2':continuation, 'A-wide':coverage}


def configuration():
    cfg = run.read_json(ROOT/CONFIG)
    require(cfg['workers']==20 and cfg['bootstrap']==5000 and cfg['bootstrap_seed']==2026093023,
            'fixed diagnostic budget changed')
    require(all(cfg[k] is v for k,v in dict(train_only=True, model_update=False,
                solver_calls=False, formal_ttf=False, automatic_promotion=False).items()), 'diagnostic scope')
    require(set(cfg['batches'])==set(cfg['models'])==set(SOURCES), 'fixed batch identities')
    return cfg


def prepare():
    cfg = configuration()
    out = ROOT/cfg['output']
    require(not out.exists(), 'existing output; verify/resume instead')
    inputs, batches = {}, {}
    parent = None
    def register(path):
        inputs[path.relative_to(ROOT).as_posix()] = run.sha256_file(path)
    for arm, module in SOURCES.items():
        reg, source = module.verify()
        require(source.relative_to(ROOT).as_posix()==cfg['batches'][arm]['source'], 'batch path')
        update = run.check_seal(run.read_json(source/'update.completion.json'))
        require(update['binding']==reg['binding'] and update['updated'], 'saved update identity')
        require(reg['config']['parent_sha256']==cfg['parent_sha256'], 'different behavior parent')
        parent_path = ROOT/reg['config']['parent_model']
        require(run.sha256_file(parent_path)==cfg['parent_sha256'], 'parent bytes changed')
        if parent is not None:
            require(parent==reg['parent'], 'different parent tensors')
        parent = reg['parent']
        spec = cfg['models'][arm]
        model = run.read_json(ROOT/spec['path'])
        require(run.sha256_file(ROOT/spec['path'])==spec['sha256']==update['model_sha256'] and
                raw.validate_bundle(model)==update['policy_sha256'] and
                model['parent_policy']==raw.validate_bundle(parent), 'same-parent model lineage')
        pairs = list(module.audited_rows(reg,source,'train'))
        require(len(pairs)==cfg['batches'][arm]['episodes'] and
                len({r['map_id'] for _,r in pairs})==cfg['batches'][arm]['maps'], 'batch coverage')
        credit_path = source/'credit.json' if arm=='A-wide' else source/'update.completion.json'
        cs = run.check_seal(run.read_json(credit_path))['coefficients']
        lookup = {c['episode_id']:c for c in cs}
        require(len(lookup)==len(pairs), 'credit completeness')
        jobs, checked_maps = [], set()
        for j,r in pairs:
            require(j['split']==r['split']=='train' and j['comparison_arm']=='parent', 'heldout or changed policy')
            c = lookup[r['episode_id']]
            torch_check = c['coefficient']!=0 and r['map_id'] not in checked_maps
            if torch_check:
                checked_maps.add(r['map_id'])
            jobs.append(dict(batch=arm, job=j, coefficient=c, torch_check=torch_check,
                             task_variant=j['case']['task_variant']))
            folder = continuation.old.previous.folder(j)
            register(folder/'result.json')
            for name in r['files']:
                register(run.contained_file(folder,name,field='frozen episode input'))
        maps = {j['case']['files']['map_file'] for j,_ in pairs}
        map_hashes = sorted({run.sha256_file(ROOT/p) for p in maps})
        require(len(map_hashes)==len(maps), 'duplicated map content')
        batches[arm] = dict(binding=reg['binding'],jobs=jobs, map_hashes=map_hashes,
                           gradient_norm=update['diagnostic']['gradient_norm'])
        for p in (parent_path,ROOT/spec['path'],source/'registration.json',source/'train.audit.json',
                  source/'train.complete.json',source/'update.completion.json',credit_path):
            register(p)
    require(not set(batches['A2']['map_hashes'])&set(batches['A-wide']['map_hashes']), 'cross-batch maps overlap')
    for p in (ROOT/CONFIG, Path(__file__),Path(diagnostic.__file__),
              ROOT/'docs/SA_COMPLETION_CREDIT_TRANSFER_PROTOCOL_ZH.md',
              ROOT/'tests/evaluation/test_sa_completion_credit_transfer.py'):
        register(p)
    protected = ['build/sa-frozen-confirmation-ttf-v1/report.json',
                 'build/sa-completion-map-coverage-v1/report.json',
                 'build/linux/sa-wall-clock-v1/lns2_env.cpython-310-x86_64-linux-gnu.so']
    for p in protected:
        register(ROOT/p)
    body = dict(schema=cfg['schema'],config=cfg,inputs=inputs,batches=batches,
                parent_path=continuation.configuration()['parent_model'],
                source_commit=run.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                posthoc=True,no_training=True,no_solver_calls=True,no_ttf=True)
    body['binding'] = run.json_fingerprint(body)
    with continuation.old.recovery.strict_lock(out,body['binding'],'credit-transfer-prepare'):
        run.once(out/'registration.json',run.sealed(body))
    return dict(binding=body['binding'],train_episodes=sum(len(b['jobs']) for b in batches.values()),
                workers=20,no_training=True,no_solver_calls=True)


def verify():
    cfg = configuration()
    out = ROOT/cfg['output']
    r = run.check_seal(run.read_json(out/'registration.json'))
    require(r['config']==cfg and r['binding']==run.json_fingerprint(
        {k:v for k,v in r.items() if k not in ('binding','integrity')}), 'registration changed')
    for name,sha in r['inputs'].items():
        require(run.sha256_file(run.contained_file(ROOT,name,field='audit input'))==sha,
                'changed input: '+name)
    for module in SOURCES.values():
        module.verify()
    return r,out


def worker(item):
    job, coefficient = item['job'],item['coefficient']
    parent = continuation.old.previous.load_model(job)
    row,pack = continuation.old.previous.extract_worker(job)
    require(row['split']=='train' and row['episode_id']==coefficient['episode_id'], 'non-train or wrong credit')
    require(pack is not None, 'empty source episode')
    score,logs = diagnostic.trajectory_score(parent,pack)
    chosen = np.arange(len(pack['selected']))
    delta = {}
    for arm,spec in item['models'].items():
        path = run.contained_file(ROOT,spec['path'],field='frozen child')
        require(run.sha256_file(path)==spec['sha256'], 'changed child')
        other,_ = diagnostic.log_probabilities(run.read_json(path),pack)
        delta[arm] = float(coefficient['coefficient'] *
                          np.sum(other[chosen,pack['selected']]-logs[chosen,pack['selected']]))
    result = dict(binding=item['binding'], batch=item['batch'], episode_id=row['episode_id'],
        job_id=job['job_id'],map_id=row['map_id'],pair_id=row['pair_id'],replica=row['replica'],
        success=row['success'],decisions=row['decisions'],task_variant=item['task_variant'],
        coefficient=coefficient['coefficient'],episode_weight=coefficient['episode_weight'],
        loss_gradient=(-coefficient['coefficient']*score).tolist(),surrogate_delta=delta,
        credit_row={k:row[k] for k in ('episode_id','pair_id','map_id','replica','split','status','success',
                    'policy_sha256','initial_fingerprint','rng_stream_id','decisions','steps',
                    'stop','generated','final_conflicts')},
        torch_check=item['torch_check'])
    if item['torch_check']:
        result['_pack'] = pack
    return result


def torch_check(parent, row, pack):
    import torch
    from experiments.sa_raw_residual_update import log_distribution
    require(torch.__version__==parent['base']['training_library'], 'frozen Torch version')
    model = raw.torch_correction(parent)
    logs = log_distribution(model,pack)
    selected = logs[torch.arange(len(pack['selected'])),torch.as_tensor(pack['selected'])]
    score = np.concatenate([g.detach().numpy().ravel() for g in torch.autograd.grad(selected.sum(),tuple(model.parameters()))])
    expected = -row['coefficient']*score
    error = float(np.max(np.abs(np.asarray(row['loss_gradient'])-expected)))
    require(error<=1e-11, 'independent Torch score mismatch')
    return error


def shard_path(out, batch, job_id):
    return out/'episodes'/batch/(job_id+'.json')


def verify_shard(doc, item):
    require(doc['binding']==item['binding'] and doc['batch']==item['batch'] and
            doc['job_id']==item['job']['job_id'] and doc['episode_id']==item['coefficient']['episode_id'] and
            doc['coefficient']==item['coefficient']['coefficient'] and doc['torch_check']==item['torch_check'],
            'stale shard')
    if item['torch_check']:
        require(0<=doc['torch_max_error']<=1e-11, 'missing Torch proof')


def analyze(resume=False):
    import torch
    torch.set_num_threads(1)
    r,out = verify()
    require(not (out/'report.json').exists(), 'completed report; verify instead')
    parent = run.read_json(ROOT/r['parent_path'])
    start = time.monotonic()
    items = [dict(j,binding=r['binding'],models=r['config']['models']) for b in r['batches'].values() for j in b['jobs']]
    rows,pending = [],[]
    with continuation.old.recovery.strict_lock(out,r['binding'],'credit-transfer-analysis'):
        run.write_json(out/'run_status.json',dict(status='running',binding=r['binding']))
        for item in items:
            path = shard_path(out,item['batch'],item['job']['job_id'])
            if path.exists():
                require(resume, 'partial output; use resume after checking interruption')
                doc = run.check_seal(run.read_json(path))
                verify_shard(doc,item)
                rows.append(doc)
            else:
                pending.append(item)
        try:
            with ProcessPoolExecutor(max_workers=r['config']['workers']) as pool:
                futures = {pool.submit(worker,item):item for item in pending}
                for future in as_completed(futures):
                    item = futures[future]
                    doc = future.result()
                    pack = doc.pop('_pack',None)
                    if pack is not None:
                        doc['torch_max_error'] = torch_check(parent,doc,pack)
                    verify_shard(doc,item)
                    run.once(shard_path(out,item['batch'],item['job']['job_id']),run.sealed(doc))
                    rows.append(doc)
                    progress = dict(phase='read_replay_score',done=len(rows),total=len(items),
                                    elapsed_seconds=time.monotonic()-start)
                    run.write_json(out/'progress.json',progress)
                    if len(rows)%12==0 or len(rows)==len(items):
                        print(json.dumps(progress),flush=True)
            result = calculate(r,rows)
            result['elapsed_seconds'] = time.monotonic()-start
            result['episode_files'] = {p.relative_to(out).as_posix():run.sha256_file(p)
                                      for p in sorted((out/'episodes').rglob('*.json'))}
            require(len(result['episode_files'])==len(items), 'missing or extra shard')
            verify()
            run.once(out/'report.json',run.sealed(result))
            run.write_json(out/'run_status.json',dict(status='complete',binding=r['binding'],decision=result['decision']))
        except BaseException as error:
            run.write_json(out/'run_status.json',dict(status='failed_or_interrupted',binding=r['binding'],
                           error=type(error).__name__+': '+str(error)))
            raise
    return {k:v for k,v in result.items() if k not in ('batches','episode_files')}


def calculate(r,rows):
    batches = {arm:sorted([x for x in rows if x['batch']==arm],key=lambda x:x['job_id']) for arm in SOURCES}
    gradients = {}
    for arm,rr in batches.items():
        source = r['batches'][arm]
        require(len(rr)==len(source['jobs']), 'incomplete batch')
        expected = {j['job']['pair_id']:j['job']['case']['map_id'] for j in source['jobs']}
        cs = continuation.objective.coefficients([x['credit_row'] for x in rr],objective='completion',
            policy_sha256=raw.validate_bundle(run.read_json(ROOT/r['parent_path'])),
            expected_groups=expected,replicas=4,node_budget=25000000)
        lookup = {c['episode_id']:c for c in cs}
        require(all(x['coefficient']==lookup[x['episode_id']]['coefficient'] and
                    x['episode_weight']==lookup[x['episode_id']]['episode_weight'] for x in rr), 'credit rederivation mismatch')
        gradients[arm] = diagnostic.loss_gradient(rr)
    parent = run.read_json(ROOT/r['parent_path'])
    summaries = {}
    for arm,rr in batches.items():
        other = next(x for x in SOURCES if x!=arm)
        summaries[arm] = diagnostic.summarize(rr,parent,run.read_json(ROOT/r['config']['models'][arm]['path']),
            r['batches'][arm]['gradient_norm'],other_direction=-gradients[other]/np.linalg.norm(gradients[other]))
    transfer = diagnostic.transfer(batches['A2'],batches['A-wide'],samples=r['config']['bootstrap'],
                                    seed=r['config']['bootstrap_seed'])
    return dict(schema=r['schema'],binding=r['binding'],batches=summaries,transfer=transfer,
        torch_checked_episodes=sum(x['torch_check'] for x in rows),
        torch_max_error=max(x.get('torch_max_error',0.) for x in rows),
        no_training=True,no_solver_calls=True,no_ttf=True,no_promotion=True,
        decision=transfer['decision'])


def verify_report():
    r,out = verify()
    report = run.check_seal(run.read_json(out/'report.json'))
    require(report['binding']==r['binding'] and report['no_training'] and report['no_solver_calls'] and
            report['no_ttf'] and report['no_promotion'], 'report scope')
    expected = {shard_path(out,j['batch'],j['job']['job_id']).relative_to(out).as_posix()
                for b in r['batches'].values() for j in b['jobs']}
    require(set(report['episode_files'])==expected, 'report coverage')
    for name,sha in report['episode_files'].items():
        require(run.sha256_file(run.contained_file(out,name,field='analysis shard'))==sha, 'changed shard: '+name)
    return dict(verified=True,episodes=len(expected),report_sha256=run.sha256_file(out/'report.json'),
                decision=report['decision'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','analyze','verify'))
    p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    result = analyze(args.resume) if args.phase=='analyze' else prepare() if args.phase=='prepare' else verify_report()
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
