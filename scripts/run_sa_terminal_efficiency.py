"""Frozen-parent, paired complete-episode objective experiment. No formal TTF."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_onpolicy as run
from scripts import run_sa_raw_residual as previous
from scripts import collect_sa_second_batch as batch
from scripts import recover_sa_onpolicy as recovery
from experiments import sa_raw_residual_actor as raw
from experiments import sa_terminal_efficiency as objective

CONFIG = 'configs/sa_terminal_efficiency.json'
ARMS = ('parent', *objective.OBJECTIVES)
require = run.require


def configuration():
    cfg = run.read_json(ROOT/CONFIG)
    require(cfg['workers'] == 20 and cfg['max_decisions'] is None and cfg['node_budget'] == 25000000,
            'fixed workers and uncapped work budget')
    require(cfg['work_bonus'] == objective.BONUS and cfg['maximum_updates_per_arm'] == 1 and
            cfg['train_replicas'] == 4 and cfg['comparison_replicas'] == 2 and
            not cfg['formal_ttf'] and not cfg['automatic_promotion'], 'fixed objective experiment')
    require(cfg['phase'] != cfg['comparison_phase'], 'training/evaluation random streams')
    return cfg


def schedule(conditions, initial, phase, replicas, arms):
    require(len(conditions) == 24 and len({c['pair_id'] for c in conditions}) == 24, '24 unique conditions')
    require(set(initial) == {c['pair_id'] for c in conditions}, 'initial fingerprint coverage')
    return [dict(c, phase=phase, replica=r, comparison_arm=arm, arm='trained_actor',
                 expected_initial=initial[c['pair_id']],
                 job_id=run.json_fingerprint([phase, c['pair_id'], r, arm])[:24])
            for c in conditions for r in range(replicas) for arm in arms]


def prepare():
    cfg = configuration()
    out = ROOT/cfg['output']
    require(not out.exists(), 'existing experiment; verify/resume rather than overwrite')
    src, ev = ROOT/cfg['source'], ROOT/cfg['evaluation_source']
    old = run.check_seal(run.read_json(src/'execution_registration.json'))
    update = run.check_seal(run.read_json(src/'update.json'))
    parent = run.read_json(ROOT/cfg['parent_model'])
    require(run.sha256_file(ROOT/cfg['parent_model']) == cfg['parent_sha256'] == update['model_sha256'] and
            raw.validate_bundle(parent) == update['policy_sha256'] and parent['iteration'] == 1, 'frozen raw-1 identity')
    e = run.check_seal(run.read_json(ev/'registration.json'))
    cases = run.check_seal(run.read_json(ev/'cases.json'))
    qual = run.check_seal(run.read_json(ev/'qualification.json'))
    require(cases['binding'] == qual['binding'] == e['binding'] and qual['collection_allowed'], 'evaluation source binding')
    train = old['conditions']
    evaluate = [dict(case=c, pair_id=f"{c['task_id']}-s{s}", solver_seed=s, split='development_holdout')
                for c in cases['cases'] for s in c['solver_seeds']]
    require(all(c['split'] == 'train' for c in train) and len(train) == len(evaluate) == 24, 'train/eval coverage')
    for field in ('map_id', 'task_id'):
        require(not {c['case'][field] for c in train} & {c['case'][field] for c in evaluate}, 'train/eval leak')
    initial = {j['pair_id']:j['expected_initial'] for j in old['jobs']}
    eval_initial = {r['pair_id']:r['initial_fingerprint'] for r in qual['rows']}
    phases = {j['phase'] for j in old['jobs']+old['comparison_jobs']} | {e['config']['phase']}
    require(not {cfg['phase'],cfg['comparison_phase']} & phases, 'reused source random stream')
    plans = dict(train=old['source_plan'], comparison=e['source_plan'])
    inputs = {}
    def register(path, expected=None):
        path = Path(path)
        name = path.relative_to(ROOT).as_posix()
        digest = run.sha256_file(path)
        require(expected is None or digest == expected, 'source input changed: '+name)
        inputs[name] = digest
    for c in train+evaluate:
        source_inputs = old['inputs'] if c['split'] == 'train' else cases['files']
        for name in c['case']['files'].values():
            register(ROOT/name, source_inputs[name])
    for plan in plans.values():
        require(plan['template']['environment']['max_repair_iterations'] == 0, 'native iteration cap')
        register(ROOT/plan['native_file'], plan['config']['native_sha256'])
    for path in [ROOT/CONFIG, ROOT/cfg['parent_model'], src/'execution_registration.json', src/'update.json',
                 ev/'registration.json', ev/'cases.json', ev/'qualification.json',
                 ROOT/'docs/SA_TERMINAL_EFFICIENCY_PROTOCOL_ZH.md',
                 ROOT/'tests/evaluation/test_sa_terminal_efficiency.py']:
        register(path)
    for directory in ('scripts', 'experiments', 'lns2_selector'):
        for path in (ROOT/directory).rglob('*.py'):
            register(path)
    for path in (ROOT/'artifacts/initlns-closed-loop-controller-v2').glob('*.json'):
        register(path)
    body = dict(schema=cfg['schema'],config=cfg,inputs=inputs,plans=plans,parent=parent,
                train_jobs=schedule(train,initial,cfg['phase'],4,['parent']),
                comparison_jobs=schedule(evaluate,eval_initial,cfg['comparison_phase'],2,ARMS),
                source_commit=run.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                no_ttf=True,no_promotion=True,evaluation_role='previously_viewed_map_disjoint_development')
    body['binding'] = run.json_fingerprint(body)
    with recovery.strict_lock(out,body['binding'],'terminal-prepare'):
        run.once(out/'registration.json',run.sealed(body))
    return dict(prepared=True,train=96,comparison=144,workers=20,max_decisions=None,no_ttf=True,binding=body['binding'])


def verify():
    cfg = configuration()
    out = ROOT/cfg['output']
    reg = run.check_seal(run.read_json(out/'registration.json'))
    require(reg['config'] == cfg and reg['binding'] == run.json_fingerprint(
        {k:v for k,v in reg.items() if k not in ('binding','integrity')}), 'registration changed')
    for name,sha in reg['inputs'].items():
        require(run.sha256_file(run.contained_file(ROOT,name,field='terminal input')) == sha, 'changed input: '+name)
    return reg,out


def model_spec(reg, arm):
    cfg = reg['config']
    if arm == 'parent':
        return dict(path=cfg['parent_model'],sha256=cfg['parent_sha256'],policy_sha256=raw.validate_bundle(reg['parent']))
    require(arm in objective.OBJECTIVES, 'unknown model arm')
    out = ROOT/cfg['output']
    result = run.check_seal(run.read_json(out/f'update.{arm}.json'))
    path = out/f'models/{arm}.json'
    require(result['binding'] == reg['binding'] and result['updated'], 'unregistered model update')
    bundle = run.read_json(path)
    require(run.sha256_file(path) == result['model_sha256'] and raw.validate_bundle(bundle) == result['policy_sha256'] and
            bundle['parent_policy'] == raw.validate_bundle(reg['parent']) and bundle['base'] == reg['parent']['base'] and
            bundle['iteration'] == reg['parent']['iteration']+1, 'model lineage/identity')
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=result['model_sha256'],policy_sha256=result['policy_sha256'])


def jobs_for(reg,lane):
    cfg = reg['config']
    jobs = []
    for job in reg[lane+'_jobs']:
        arm = job['comparison_arm']
        p = deepcopy(reg['plans'][lane])
        p['config']['output'] = cfg['output']+f'/{lane}/{arm}'
        p['proposal'].update({k:cfg[k] for k in ('max_decisions','node_budget','pp_safety_seconds','episode_safety_seconds','process_fuse_seconds')})
        p['proposal']['decision_feature_reference'] = 256
        spec = model_spec(reg,arm)
        jobs.append(dict(job,plan=p,model=spec,iteration=1 if arm=='parent' else 2,
                         parent_pid=os.getpid(),comparison_binding=reg['binding']))
    return jobs


def collect(lane,resume=False):
    reg,out = verify()
    if lane == 'comparison':
        for platform in ('windows','wsl'):
            proof = run.check_seal(run.read_json(out/f'parity.{platform}.json'))
            require(proof['binding'] == reg['binding'] and proof['exact_choices'] and
                    proof['models'] == {a:model_spec(reg,a) for a in objective.OBJECTIVES}, 'portable parity missing')
    jobs = jobs_for(reg,lane)
    with recovery.strict_lock(out,reg['binding'],'terminal-'+lane):
        done = batch.execute_batches(reg,out,jobs,previous.worker,previous.folder,previous.read_result,
                                     lane,resume,reg['config']['process_fuse_seconds'])
    return dict(complete=done,episodes=len(jobs),no_ttf=True)


def audited_rows(reg,out,lane):
    complete = run.check_seal(run.read_json(out/f'{lane}.complete.json'))
    audit = run.check_seal(run.read_json(out/f'{lane}.audit.json'))
    jobs = jobs_for(reg,lane)
    hashes = {r['job_id']:r['result_sha256'] for r in audit['results'] if r['status']=='ok'}
    require(complete['binding'] == audit['binding'] == reg['binding'] and complete['jobs'] == len(jobs) and
            len(hashes) == len(audit['results']) == len(jobs) and set(hashes) == set(complete['files']), 'audit coverage')
    for j in jobs:
        require(run.sha256_file(previous.folder(j)/'result.json') == hashes[j['job_id']] == complete['files'][j['job_id']], 'stale audit')
        row = previous.read_result(j)
        require(row['status'] == 'ok', 'censored batch cannot train or evaluate')
        yield j,row


def audit(lane):
    from experiments.repair_collection import _run_jobs
    reg,out = verify()
    complete = run.check_seal(run.read_json(out/f'{lane}.complete.json'))
    jobs = jobs_for(reg,lane)
    require(complete['binding'] == reg['binding'] and complete['jobs'] == len(jobs), 'incomplete collection')
    for j in jobs:
        require(complete['files'][j['job_id']] == run.sha256_file(previous.folder(j)/'result.json') and
                previous.read_result(j)['status'] == 'ok', 'changed/censored collection')
    with recovery.strict_lock(out,reg['binding'],'terminal-audit-'+lane):
        rows = _run_jobs(previous.audit_worker,jobs,20,phase='audit-'+lane,output_root=out/('audit-progress-'+lane),
                         run_fingerprint=reg['binding'],timeout_seconds=960.)
        require(len(rows) == len(jobs) and all(r['status']=='ok' for r in rows), 'full audit failed')
        run.once(out/f'{lane}.audit.json',run.sealed(dict(binding=reg['binding'],results=rows)))
    return dict(audited=len(rows),lane=lane)


def train():
    import torch
    from experiments.sa_raw_residual_update import guarded_update
    torch.set_num_threads(1)
    reg,out = verify()
    parent = reg['parent']
    require(torch.__version__ == parent['base']['training_library'], 'frozen Torch version')
    with recovery.strict_lock(out,reg['binding'],'terminal-two-updates'):
        require(not any((out/f'update.{a}.json').exists() or (out/f'models/{a}.json').exists()
                        for a in objective.OBJECTIVES), 'one update per arm; partial update requires inspection')
        jobs = [j for j,_ in audited_rows(reg,out,'train')]
        require(len(jobs) == 96 and all(j['split']=='train' for j in jobs), 'train-only update')
        rows,packs = [],[]
        with ProcessPoolExecutor(max_workers=20) as pool:
            for row,pack in pool.map(previous.extract_worker,jobs):
                rows.append(row)
                packs.append(pack)
                if len(rows)%12 == 0:
                    print(json.dumps(dict(phase='extract',episodes=len(rows),total=96)),flush=True)
        results = {}
        for arm in objective.OBJECTIVES:
            cs = objective.coefficients(rows,objective=arm,policy_sha256=raw.validate_bundle(parent),
                expected_groups={j['pair_id']:j['case']['map_id'] for j in jobs},replicas=4,node_budget=reg['config']['node_budget'])
            lookup = {c['episode_id']:c for c in cs}
            active = [dict(p,weight=lookup[r['episode_id']]['episode_weight'],coefficient=lookup[r['episode_id']]['coefficient'])
                      for r,p in zip(rows,packs) if p is not None]
            proposed,diagnostic = guarded_update(parent,active,reg['binding'])
            changed = None if proposed is None else {
                k:sum(a!=b for a,b in zip(flat(parent['correction'][k]),flat(proposed['correction'][k])))
                for k in ('w1','b1','w2','b2')}
            result = dict(binding=reg['binding'],objective=arm,updated=proposed is not None,diagnostic=diagnostic,
                coefficients=cs,credited_episodes=sum(c['coefficient']!=0 for c in cs),
                credited_maps=sorted({r['map_id'] for r in rows if lookup[r['episode_id']]['coefficient']!=0}),
                episodes=len(rows),successes=sum(r['success'] for r in rows),decisions=sum(r['decisions'] for r in rows),
                changed_parameters=changed,train_audit_sha256=run.sha256_file(out/'train.audit.json'),no_ttf=True,no_promotion=True)
            if proposed is not None:
                path = out/f'models/{arm}.json'
                run.once(path,proposed)
                result.update(model_sha256=run.sha256_file(path),policy_sha256=raw.validate_bundle(proposed))
            run.once(out/f'update.{arm}.json',run.sealed(result))
            results[arm] = {k:v for k,v in result.items() if k!='coefficients'}
            print(json.dumps(dict(phase='updated',**results[arm])),flush=True)
    return results


def flat(values):
    return [v for row in values for v in row] if isinstance(values[0],list) else values


def parity(platform):
    reg,out = verify()
    specs = {a:model_spec(reg,a) for a in objective.OBJECTIVES}
    fixtures = []
    if platform == 'windows':
        import torch
        torch.set_num_threads(1)
        for j,row in audited_rows(reg,out,'train'):
            for event in run.trace_read(previous.folder(j)):
                if event['decision'] not in {0,row['decisions']//2,row['decisions']-1}:
                    continue
                fixtures.append(dict(ids=event['candidate_ids'],anchor=event['anchor_id'],features=event['features'],
                                     draws=[event['selection_draw'],*[i/16 for i in range(16)]]))
        for arm,spec in specs.items():
            bundle = run.read_json(ROOT/spec['path'])
            model = raw.torch_correction(bundle)
            for f in fixtures:
                with torch.no_grad():
                    ps = raw.torch_log_distribution(model,bundle,f['ids'],f['anchor'],f['features']).exp().tolist()
                f.setdefault('probabilities',{})[arm] = dict(zip(sorted(f['ids']),ps))
    else:
        run.native_runtime(reg['plans']['train'])
        ref = run.check_seal(run.read_json(out/'parity.windows.json'))
        require(ref['binding'] == reg['binding'] and ref['models'] == specs, 'parity identity')
        fixtures = ref['fixtures']
    require(fixtures, 'empty parity fixtures')
    error,choices = 0.,0
    for arm,spec in specs.items():
        actor = raw.RawResidualActor(run.read_json(ROOT/spec['path']))
        for f in fixtures:
            ps = actor.probabilities(f['ids'],f['anchor'],f['features'])
            expected = f['probabilities'][arm]
            error = max(error,max(abs(ps[c]-expected[c]) for c in ps))
            require(error <= 1e-12, 'portable probability mismatch')
            for draw in f['draws']:
                require(run.select_with_draw(ps,draw) == run.select_with_draw(expected,draw), 'portable choice mismatch')
                choices += 1
    result = dict(binding=reg['binding'],models=specs,fixtures_count=len(fixtures),choices=choices,
                  max_error=error,exact_choices=True)
    run.once(out/f'parity.{platform}.json',run.sealed(dict(result,fixtures=fixtures) if platform=='windows' else result))
    return result


def report():
    reg,out = verify()
    rows = [dict(r,comparison_arm=j['comparison_arm']) for j,r in audited_rows(reg,out,'comparison')]
    cfg = reg['config']
    contrasts = {a:objective.contrast(rows,'parent',a,bootstrap=cfg['bootstrap'],seed=cfg['bootstrap_seed'])
                 for a in objective.OBJECTIVES}
    contrasts['objective_increment'] = objective.contrast(rows,'completion','completion_work',
        bootstrap=cfg['bootstrap'],seed=cfg['bootstrap_seed'])
    signals = {}
    for arm in objective.OBJECTIVES:
        c = contrasts[arm]
        work = c['metrics']['generated']['change_percent']
        signals[arm] = c['challenger_success'] >= c['baseline_success'] and (
            c['challenger_success'] > c['baseline_success'] or (work is not None and work <= -5.))
    result = dict(binding=reg['binding'],episodes=rows,contrasts=contrasts,development_signal=signals,
        successes={a:sum(r['success'] for r in rows if r['comparison_arm']==a) for a in ARMS},
        decision='development_signal_not_ttf' if any(signals.values()) else 'no_development_gain_stop_this_update',
        no_ttf=True,no_promotion=True,independent_generalization=False,
        comparison_audit_sha256=run.sha256_file(out/'comparison.audit.json'),
        update_sha256={a:run.sha256_file(out/f'update.{a}.json') for a in objective.OBJECTIVES})
    run.once(out/'report.json',run.sealed(result))
    run.write_json(out/'run_status.json',dict(status='complete',binding=reg['binding'],decision=result['decision']))
    return {k:v for k,v in result.items() if k!='episodes'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','verify','collect','audit','train','parity','compare','report','stop'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--lane',choices=('train','comparison'),default='train')
    parser.add_argument('--platform',choices=('windows','wsl'))
    args = parser.parse_args()
    if args.phase == 'verify': result = dict(verified=True,binding=verify()[0]['binding'])
    elif args.phase in ('collect','compare'): result = collect('train' if args.phase=='collect' else 'comparison',args.resume)
    elif args.phase == 'audit': result = audit(args.lane)
    elif args.phase == 'parity':
        require(args.platform is not None,'platform required')
        result = parity(args.platform)
    elif args.phase == 'stop':
        run.write_json(ROOT/configuration()['output']/'STOP_AFTER_BATCH',dict(requested=True))
        result = dict(stop_after_current_batch=True)
    else: result = globals()[args.phase]()
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
