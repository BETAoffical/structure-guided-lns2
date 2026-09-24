"""Isolated raw actor: native admission, one new terminal batch and paired comparison."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_onpolicy as run
from scripts import collect_sa_second_batch as batch
from scripts import recover_sa_onpolicy as recovery
from scripts import check_sa_raw_residual_contract as contract
from experiments import sa_raw_residual_actor as raw
from experiments import sa_uncapped_runtime as runtime
from experiments import sa_uncapped_training_contract as credit
from experiments.sa_raw_residual_runtime import actor_scope

CONFIG = 'configs/sa_raw_residual_runtime.json'
require = run.require


def configuration():
    cfg = run.read_json(ROOT/CONFIG)
    expected = dict(phase='raw-terminal-train-20260925', comparison_phase='raw-terminal-compare-20260925',
        train_replicas=4, comparison_replicas=2, conditions=24, workers=20, max_decisions=None,
        node_budget=25000000, pp_safety_seconds=20., episode_safety_seconds=900., process_fuse_seconds=960.,
        maximum_updates=1, formal_ttf=False, heldout_evaluation=False, automatic_promotion=False)
    require(all(cfg[k] == v for k,v in expected.items()), 'fixed raw experiment scope')
    return cfg


def schedule(conditions, initial, phase, replicas, iterations):
    require(len(conditions) == 24 and all(c['split'] == 'train' for c in conditions), 'Train coverage')
    return [dict(c, phase=phase, replica=r, iteration=i, arm='trained_actor',
        expected_initial=initial[c['pair_id']],
        job_id=run.json_fingerprint([phase, c['pair_id'], r, i])[:24])
        for c in conditions for r in range(replicas) for i in iterations]


def prepare():
    cfg = configuration()
    out = ROOT/cfg['output']
    require(not out.exists(), 'existing output; inspect/resume')
    prior = contract.verify()
    require(run.sha256_file(contract.OUT/'contract.complete.json') ==
        'ff4acded0d321d06ffc2eb14ed534857897a4ed0bc4b36a1cfc53c44e6139955', 'frozen completed contract')
    source = ROOT/cfg['source']
    old = run.check_seal(run.read_json(source/'registration.json'))
    report = run.check_seal(run.read_json(source/'report.json'))
    require(run.sha256_file(source/'report.json') ==
        '869ae98b2fbbd3b7d9f1aa014f46576a11b8e9599fd118d91b206d4cbea628c1', 'source report SHA')
    cooling_path = ROOT/'build/sa-budget-cooling-probe-v1/comparison_registration.json'
    source_plan = run.check_seal(run.read_json(cooling_path))['source_plan']
    require(source_plan['binding'] == old['scientific_binding'] and
        source_plan['template']['environment']['max_repair_iterations'] == 0, 'uncapped native plan')
    require(prior['prototype']['base_policy_sha256'] == old['config']['models'][batch.ARMS[1]]['policy_sha256'], 'frozen parent')
    rows = [r for r in report['episodes'] if r['comparison_arm'] == batch.ARMS[1]]
    require(len(rows) == 96 and all(r['split'] == 'train' and r['status'] == 'ok' for r in rows), 'source Train only')
    initial = {}
    for row in rows:
        initial.setdefault(row['pair_id'], row['initial_fingerprint'])
        require(initial[row['pair_id']] == row['initial_fingerprint'], 'source pairing')
    controls, maps = [], set()
    for row in sorted(rows, key=lambda r:(r['decisions'], r['job_id'])):
        if row['decisions'] == 0 or row['map_id'] in maps:
            continue
        j = next(j for j in old['jobs'] if j['job_id'] == row['job_id'])
        path = source/'runs'/batch.ARMS[1]/j['phase']/j['job_id']
        controls.append(dict(j, iteration=0, expected_initial=row['initial_fingerprint'],
            original_folder=path.relative_to(ROOT).as_posix()))
        maps.add(row['map_id'])
        if len(controls) == 2:
            break
    require(len(controls) == 2, 'two different-map full-episode controls')
    jobs = schedule(old['conditions'], initial, cfg['phase'], 4, [0])
    comparisons = schedule(old['conditions'], initial, cfg['comparison_phase'], 2, [0,1])
    require(len(jobs) == len(comparisons) == 96, 'registered counts')
    require(not {j['phase'] for j in jobs+comparisons} & {j['phase'] for j in old['jobs']}, 'fresh random streams')
    inputs = dict(prior['inputs'])
    for path in [ROOT/CONFIG, cooling_path, source/'registration.json', source/'report.json',
                 contract.OUT/'prototype.initial.json', contract.OUT/'contract.complete.json',
                 ROOT/'docs/SA_RAW_RESIDUAL_RUNTIME_PROTOCOL_ZH.md',
                 ROOT/'tests/evaluation/test_sa_raw_residual_runtime.py']:
        inputs[path.relative_to(ROOT).as_posix()] = run.sha256_file(path)
    for directory in ('scripts','experiments','lns2_selector'):
        for path in (ROOT/directory).rglob('*.py'):
            inputs[path.relative_to(ROOT).as_posix()] = run.sha256_file(path)
    for c in old['conditions']:
        for name in c['case']['files'].values():
            inputs[name] = run.sha256_file(ROOT/name)
    inputs[source_plan['native_file']] = source_plan['config']['native_sha256']
    for path in (ROOT/'artifacts/initlns-closed-loop-controller-v2').glob('*.json'):
        inputs[path.relative_to(ROOT).as_posix()] = run.sha256_file(path)
    body = dict(schema=cfg['schema'], config=cfg, inputs=inputs, source_plan=source_plan,
        source_config=old['config'], conditions=old['conditions'], controls=controls, jobs=jobs,
        comparison_jobs=comparisons, prototype=prior['prototype'], no_ttf=True, no_heldout=True,
        source_commit=run.subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip())
    body['binding'] = run.json_fingerprint(body)
    with recovery.strict_lock(out, body['binding'], 'raw-prepare'):
        run.once(out/'models/raw-0.json', body['prototype'])
        run.once(out/'execution_registration.json', run.sealed(body))
    return dict(registered=True, controls=2, new_train_episodes=96, paired_episodes_if_update=96,
                updates=1, max_decisions=None, workers=20, no_ttf=True)


def verify():
    cfg = configuration()
    out = ROOT/cfg['output']
    reg = run.check_seal(run.read_json(out/'execution_registration.json'))
    require(reg['config'] == cfg and reg['binding'] == run.json_fingerprint(
        {k:v for k,v in reg.items() if k not in ('binding','integrity')}), 'registration changed')
    for name, sha in reg['inputs'].items():
        require(run.sha256_file(run.contained_file(ROOT,name,field='raw input')) == sha, 'input changed: '+name)
    require(run.read_json(out/'models/raw-0.json') == reg['prototype'], 'initial model changed')
    return reg, out


def model_spec(reg, iteration):
    out = ROOT/reg['config']['output']
    path = out/f'models/raw-{iteration}.json'
    bundle = run.read_json(path)
    if iteration == 0:
        require(bundle == reg['prototype'], 'initial raw identity')
    else:
        update = run.check_seal(run.read_json(out/'update.json'))
        require(iteration == 1 and update['binding'] == reg['binding'] and update['updated'], 'unregistered update')
        require(run.sha256_file(path) == update['model_sha256'] and raw.validate_bundle(bundle) == update['policy_sha256'], 'updated model SHA')
        require(bundle['parent_policy'] == raw.validate_bundle(reg['prototype']) and bundle['base'] == reg['prototype']['base'], 'model lineage')
    require(bundle['iteration'] == iteration, 'model iteration')
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=run.sha256_file(path), policy_sha256=raw.validate_bundle(bundle))


def augment(reg, job, lane):
    cfg = reg['config']
    p = deepcopy(reg['source_plan'])
    p['config']['output'] = cfg['output'] + '/' + lane
    p['proposal'].update({k:cfg[k] for k in ('max_decisions','node_budget','pp_safety_seconds','episode_safety_seconds','process_fuse_seconds')})
    p['proposal']['decision_feature_reference'] = 256
    return dict(job, plan=p, model=model_spec(reg,job['iteration']), parent_pid=os.getpid(), comparison_binding=reg['binding'])


def folder(job):
    return run.folder_for(ROOT/job['plan']['config']['output'], job['phase'], job)


def load_model(job):
    spec = job['model']
    path = run.contained_file(ROOT,spec['path'],field='raw worker model')
    require(run.sha256_file(path) == spec['sha256'], 'worker model bytes')
    bundle = run.read_json(path)
    require(raw.validate_bundle(bundle) == spec['policy_sha256'] and bundle['iteration'] == job['iteration'], 'worker model identity')
    return bundle


def worker(job):
    from scripts.train_sa_history_selector import die_with_parent
    die_with_parent(job['parent_pid'])
    bundle = load_model(job)
    with actor_scope(bundle, ROOT/job['plan']['config']['output'], job['plan']):
        return runtime.episode_worker(job)


def audit_worker(job):
    bundle = load_model(job)
    with actor_scope(bundle, ROOT/job['plan']['config']['output'], job['plan']):
        return runtime.audit_worker(job)


def read_result(job):
    row = run.result_read(folder(job), job['plan'])
    require(row['execution_binding'] == job['comparison_binding'] and row['policy_sha256'] == job['model']['policy_sha256'], 'execution/model identity')
    require(row['initial_fingerprint'] == job['expected_initial'] and row['map_id'] == job['case']['map_id'], 'task identity')
    require(all(row[k] == job[k] for k in ('job_id','pair_id','replica','split','arm')), 'job identity')
    runtime.validate_terminal(row, job['plan']['proposal'])
    return row


def controls(resume=False):
    reg,out = verify()
    jobs = [augment(reg,j,'controls') for j in reg['controls']]
    with recovery.strict_lock(out,reg['binding'],'raw-controls'):
        if not batch.execute_batches(reg,out,jobs,worker,folder,read_result,'controls',resume,960.):
            return dict(status='paused_or_needs_inspection')
        checks = []
        for j in jobs:
            row = read_result(j)
            old = run.result_read(ROOT/j['original_folder'], reg['source_plan'])
            require(all(old[k] == row[k] for k in ('initial_fingerprint','final_fingerprint','success','stop',
                        'generated','decisions','soc','makespan','wait_steps','rng_stream_id')), 'zero-update episode changed')
            events = list(run.trace_read(folder(j)))
            previous = list(run.trace_read(ROOT/j['original_folder']))
            require(len(events) == len(previous) == row['decisions'], 'control coverage')
            for a,b in zip(previous,events):
                require({k:v for k,v in a.items() if k not in ('policy_sha256','metrics')} ==
                        {k:v for k,v in b.items() if k not in ('policy_sha256','metrics')}, 'zero-update action/path/features changed')
            checks.append(dict(job_id=j['job_id'],decisions=len(events),audit=audit_worker(j)))
        run.once(out/'controls.parity.json',run.sealed(dict(binding=reg['binding'],exact=True,results=checks)))
    return dict(exact=True,episodes=len(checks),decisions=sum(r['decisions'] for r in checks),no_ttf=True)


def admitted(reg,out,lane):
    proof = run.check_seal(run.read_json(out/'controls.parity.json'))
    require(proof['binding'] == reg['binding'] and proof['exact'] and len(proof['results']) == 2, 'controls not admitted')
    for j,r in zip(reg['controls'],proof['results']):
        job = augment(reg,j,'controls')
        require(r['job_id'] == j['job_id'] and r['audit']['result_sha256'] == run.sha256_file(folder(job)/'result.json'), 'stale controls')
        read_result(job)
    if lane == 'comparison':
        for name in ('windows','wsl'):
            parity = run.check_seal(run.read_json(out/f'parity.{name}.json'))
            require(parity['binding'] == reg['binding'] and parity['model_sha256'] == model_spec(reg,1)['sha256'] and parity['exact_choices'], 'updated portable parity missing')


def collect(lane, resume=False):
    reg,out = verify()
    admitted(reg,out,lane)
    jobs = [augment(reg,j,lane) for j in reg['jobs' if lane == 'train' else 'comparison_jobs']]
    with recovery.strict_lock(out,reg['binding'],'raw-'+lane):
        done = batch.execute_batches(reg,out,jobs,worker,folder,read_result,lane,resume,960.)
    return dict(complete=done,episodes=len(jobs),no_ttf=True)


def audited_rows(reg,out,lane):
    key = 'jobs' if lane == 'train' else 'comparison_jobs'
    complete = run.check_seal(run.read_json(out/f'{lane}.complete.json'))
    audit = run.check_seal(run.read_json(out/f'{lane}.audit.json'))
    jobs = [augment(reg,j,lane) for j in reg[key]]
    hashes = {r['job_id']:r['result_sha256'] for r in audit['results'] if r['status']=='ok'}
    require(complete['binding'] == audit['binding'] == reg['binding'] and complete['jobs'] == len(jobs), 'complete audited collection required')
    require(len(hashes) == len(audit['results']) == len(jobs) and set(hashes) == set(complete['files']), 'audit coverage')
    for j in jobs:
        require(run.sha256_file(folder(j)/'result.json') == hashes[j['job_id']] == complete['files'][j['job_id']], 'stale collection audit')
        row = read_result(j)
        require(row['status'] == 'ok', 'censored batch cannot be used')
        yield j,row


def audit(lane):
    from experiments.repair_collection import _run_jobs
    reg,out = verify()
    complete = run.check_seal(run.read_json(out/f'{lane}.complete.json'))
    jobs = [augment(reg,j,lane) for j in reg['jobs' if lane == 'train' else 'comparison_jobs']]
    require(complete['binding'] == reg['binding'] and complete['jobs'] == len(jobs), 'collection not complete')
    for j in jobs:
        require(read_result(j)['status'] == 'ok' and complete['files'][j['job_id']] == run.sha256_file(folder(j)/'result.json'), 'collection changed')
    with recovery.strict_lock(out,reg['binding'],'raw-audit-'+lane):
        rows = _run_jobs(audit_worker,jobs,20,phase='audit-'+lane,output_root=out/('audit-progress-'+lane),
                        run_fingerprint=reg['binding'],timeout_seconds=960.)
        require(len(rows) == len(jobs) and all(r['status']=='ok' for r in rows), 'full audit failed')
        run.once(out/f'{lane}.audit.json',run.sealed(dict(binding=reg['binding'],results=rows)))
    return dict(audited=len(rows),lane=lane)


def extract_worker(job):
    from experiments.sa_raw_residual_update import pack_episode
    row = read_result(job)
    events = list(run.trace_read(folder(job)))
    require(len(events) == row['decisions'], 'trajectory coverage')
    pack = pack_episode(load_model(job),events)
    if pack is None:
        require(row['success'] and row['decisions'] == 0, 'unexplained empty episode')
    reduced = [{k:e[k] for k in ('decision','policy_sha256','probabilities','selected_id','behavior_log_probability')} for e in events]
    return dict(row,steps=reduced),pack


def train():
    import torch
    from experiments.sa_raw_residual_update import guarded_update
    torch.set_num_threads(1)
    reg,out = verify()
    bundle = reg['prototype']
    require(torch.__version__ == bundle['base']['training_library'], 'frozen Torch version')
    with recovery.strict_lock(out,reg['binding'],'raw-one-update'):
        require(not (out/'update.json').exists() and not (out/'models/raw-1.json').exists(), 'one update only; partial update needs inspection')
        jobs = [j for j,_ in audited_rows(reg,out,'train')]
        rows,packs = [],[]
        with ProcessPoolExecutor(max_workers=20) as pool:
            for row,pack in pool.map(extract_worker,jobs):
                rows.append(row)
                packs.append(pack)
                if len(rows)%12 == 0:
                    print(json.dumps(dict(phase='extract',episodes=len(rows),total=96)),flush=True)
        coefficients = credit.gradient_coefficients(rows, policy_sha256=raw.validate_bundle(bundle),
            expected_groups={j['pair_id']:j['case']['map_id'] for j in reg['jobs']},replicas=4,max_decisions=None,node_budget=25000000)
        weights = {c['episode_id']:c for c in coefficients}
        active = []
        for row,pack in zip(rows,packs):
            if pack is not None:
                c = weights[row['episode_id']]
                active.append(dict(pack,weight=c['episode_weight'],coefficient=c['coefficient']))
        new,diagnostic = guarded_update(bundle,active,reg['binding'])
        result = dict(binding=reg['binding'],updated=new is not None,diagnostic=diagnostic,
            coefficients=coefficients,successes=sum(r['success'] for r in rows),episodes=len(rows),
            decisions=sum(r['decisions'] for r in rows),audit_sha256=run.sha256_file(out/'train.audit.json'),
            credited_episodes=sum(c['coefficient'] != 0 for c in coefficients),
            credited_maps=sorted({r['map_id'] for r in rows if weights[r['episode_id']]['coefficient'] != 0}),
            no_ttf=True,no_heldout=True,no_promotion=True)
        if new is not None:
            run.once(out/'models/raw-1.json',new)
            result.update(model_sha256=run.sha256_file(out/'models/raw-1.json'),policy_sha256=raw.validate_bundle(new))
        run.once(out/'update.json',run.sealed(result))
    return {k:v for k,v in result.items() if k != 'coefficients'}


def parity(label):
    reg,out = verify()
    spec = model_spec(reg,1)
    bundle = run.read_json(ROOT/spec['path'])
    actor = raw.RawResidualActor(bundle)
    fixtures = []
    if label == 'windows':
        import torch
        torch.set_num_threads(1)
        model = raw.torch_correction(bundle)
        for j,row in audited_rows(reg,out,'train'):
            for e in run.trace_read(folder(j)):
                if e['decision'] not in {0,row['decisions']//2,row['decisions']-1}:
                    continue
                with torch.no_grad():
                    ps = raw.torch_log_distribution(model,bundle,e['candidate_ids'],e['anchor_id'],e['features']).exp().tolist()
                fixtures.append(dict(ids=e['candidate_ids'],anchor=e['anchor_id'],features=e['features'],
                    probabilities=dict(zip(sorted(e['candidate_ids']),ps)),draws=[e['selection_draw'],*[i/16 for i in range(16)]]))
    else:
        run.native_runtime(reg['source_plan'])
        ref = run.check_seal(run.read_json(out/'parity.windows.json'))
        require(ref['binding'] == reg['binding'] and ref['model_sha256'] == spec['sha256'], 'parity source changed')
        fixtures = ref['fixtures']
    require(fixtures, 'empty parity')
    error,choices = 0.,0
    for f in fixtures:
        ps = actor.probabilities(f['ids'],f['anchor'],f['features'])
        error = max(error,max(abs(ps[k]-f['probabilities'][k]) for k in ps))
        require(error <= 1e-12, 'portable probability mismatch')
        for draw in f['draws']:
            require(run.select_with_draw(ps,draw) == run.select_with_draw(f['probabilities'],draw), 'portable choice mismatch')
            choices += 1
    result = dict(binding=reg['binding'],model_sha256=spec['sha256'],fixtures_count=len(fixtures),
                  choices=choices,max_error=error,exact_choices=True)
    run.once(out/f'parity.{label}.json',run.sealed(dict(result,fixtures=fixtures) if label=='windows' else result))
    return result


def report():
    from scripts.probe_sa_budget_cooling import map_bootstrap
    reg,out = verify()
    rows = [dict(row,iteration=j['iteration']) for j,row in audited_rows(reg,out,'comparison')]
    def contrast(subset):
        groups = {}
        for row in subset:
            key = (row['pair_id'],row['replica'])
            require(row['iteration'] not in groups.setdefault(key,{}), 'duplicate pair')
            groups[key][row['iteration']] = row
        wins,losses,common = [],[],[]
        for key,pair in sorted(groups.items()):
            require(set(pair)=={0,1},'missing comparison arm')
            a,b = pair[0],pair[1]
            require(all(a[k]==b[k] for k in ('initial_fingerprint','rng_stream_id','map_id')), 'unpaired comparison')
            if b['success'] and not a['success']:wins.append(list(key))
            if a['success'] and not b['success']:losses.append(list(key))
            if a['success'] and b['success']:common.append((a,b))
        return dict(pairs=len(groups),gains=wins,losses=losses,net_success=len(wins)-len(losses),
            common_success=len(common),common_totals={str(i):{k:sum(p[i][k] for p in common)
                for k in ('generated','decisions','soc','makespan','wait_steps')} for i in (0,1)})
    overall = contrast(rows)
    by_map = {m:contrast([r for r in rows if r['map_id']==m]) for m in sorted({r['map_id'] for r in rows})}
    decision = ('positive_development_signal_needs_independent_confirmation' if overall['net_success']>0 else
                'no_net_completion_gain_no_promotion')
    result = dict(binding=reg['binding'],decision=decision,overall=overall,by_map=by_map,
        bootstrap=map_bootstrap(by_map),successes={str(i):sum(r['success'] for r in rows if r['iteration']==i) for i in (0,1)},
        stops=dict(Counter(r['stop'] for r in rows)),episodes=rows,no_ttf=True,no_promotion=True,
        comparison_audit_sha256=run.sha256_file(out/'comparison.audit.json'),update_sha256=run.sha256_file(out/'update.json'))
    run.once(out/'report.json',run.sealed(result))
    run.write_json(out/'run_status.json',dict(status='complete',decision=decision,episodes=96))
    return {k:v for k,v in result.items() if k!='episodes'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','verify','controls','collect','audit','train','parity','compare','report','stop'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--lane',choices=('train','comparison'),default='train')
    parser.add_argument('--platform',choices=('windows','wsl'))
    args = parser.parse_args()
    if args.phase == 'verify':result = dict(verified=True,binding=verify()[0]['binding'])
    elif args.phase == 'controls':result = controls(args.resume)
    elif args.phase == 'collect':result = collect('train',args.resume)
    elif args.phase == 'compare':result = collect('comparison',args.resume)
    elif args.phase == 'audit':result = audit(args.lane)
    elif args.phase == 'parity':
        require(args.platform is not None,'platform required')
        result = parity(args.platform)
    elif args.phase == 'stop':
        run.write_json(ROOT/configuration()['output']/'STOP_AFTER_BATCH',dict(requested=True))
        result = dict(stop_after_current_batch=True)
    else:result = globals()[args.phase]()
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
