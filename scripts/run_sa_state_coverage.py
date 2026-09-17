"""Bounded independent-episode coverage test for frozen H32 model designs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, write_json, sha256_file, json_fingerprint
from experiments.sa_paired_completion import SCHEMA, MODEL_PARAMS, require, inspect_dataset
from experiments.sa_history_information import profile_features
from experiments.sa_completion_contract import split_diagnostics
from experiments.sa_paired_sampling import work_budget
from experiments.sa_state_coverage import choose_new_roots, merge_data, fit_models, predict, contrast, signal_gate
from scripts import run_sa_paired_completion_pilot as pilot

CONFIG = ROOT / 'configs/sa_state_coverage.json'


def freeze_fold(job):
    data, map_id, folder = job
    model = fit_models(data, map_id)
    path = Path(folder) / (map_id + '.pkl')
    require(not path.exists(), 'frozen fold exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        pickle.dump(model, stream, protocol=4)
    return dict(map_id=map_id, file=path.relative_to(ROOT).as_posix(), sha256=sha256_file(path),
                train_ids=model['train_ids'], train_maps=model['train_maps'])


def allow_preparation_output(out, resume):
    if not out.exists():
        return
    require(resume and {p.name for p in out.iterdir()} == {'coverage.json'} and
            read_json(out/'coverage.json').get('admission') is False,
            'output exists; only an unsealed failed preflight may be resumed')


def require_admission(plan):
    require(plan['admission'], 'insufficient source coverage; no solver collection or training authorized')


def prepare(resume=False):
    old_plan, old_out = pilot.verify()
    cfg = read_json(CONFIG)
    out = ROOT / cfg['output']
    allow_preparation_output(out, resume)
    require((cfg['maps'], cfg['states'], cfg['max_candidates'], cfg['trials'], cfg['horizon'], cfg['workers']) ==
            (8, 32, 4, 8, 32, 20), 'fixed coverage budget changed')
    old = read_json(old_out / 'training_index.json')
    receipt = read_json(old_out / 'training_receipt.json')
    require(receipt['binding'] == old_plan['binding'] and
            receipt['input_sha256'] == sha256_file(old_out / 'training_index.json') and
            receipt['report_sha256'] == sha256_file(old_out / 'model_report.json'), 'old training identity')
    sampling = read_json(ROOT / 'build/sa-history-selector-sampling-v2/plan.json')
    excluded = set(old_plan['excluded_episodes']) | {r['item']['job_id'] for r in old_plan['roots']} | {
        r['item']['job_id'] for r in sampling['targets']}
    reg = read_json(ROOT / cfg['source'] / 'registration.json')
    cases = {c['task_id']: c for c in reg['cases']}
    maps = {s['map_id'] for s in old['states']}
    jobs = [dict(config=cfg, item=item, map_id=cases[item['task_id']]['map_id']) for item in reg['schedule']
            if item['controller'] == 'dual16_sa' and item['job_id'] not in excluded]
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        available = [r for rows in pool.map(pilot.scan_source, jobs) for r in rows]
    roots, coverage = choose_new_roots(available, maps, excluded, cfg['seed'])
    admission = len(roots) == cfg['states'] and len(maps) == cfg['maps'] and all(c['admitted'] for c in coverage)
    pilot.save_once(out / 'coverage.json', dict(admission=admission, coverage=coverage))
    frozen = []
    if admission:
        with ProcessPoolExecutor(max_workers=min(cfg['workers'], len(maps))) as pool:
            frozen = list(pool.map(freeze_fold, [(old, m, str(out / 'old_fold_models')) for m in sorted(maps)]))
    inputs = dict(old_plan['inputs'])
    files = [CONFIG, Path(__file__), ROOT / 'experiments/sa_state_coverage.py',
             ROOT / 'tests/evaluation/test_sa_state_coverage.py', ROOT / 'docs/SA_STATE_COVERAGE_PROTOCOL_ZH.md',
             ROOT / 'experiments/sa_completion_contract.py', old_out / 'plan.json',
             old_out / 'training_index.json', old_out / 'training_receipt.json', old_out / 'model_report.json',
             ROOT / 'build/sa-history-selector-sampling-v2/plan.json', out / 'coverage.json']
    # Outcome summaries only explain coverage after selection is frozen; they
    # never enter choose_new_roots or authorize replacing an episode.
    manifest = read_json(ROOT / cfg['source'] / 'manifest.json')
    inventory = []
    for job in jobs:
        item = job['item']
        folder = ROOT / cfg['source'] / 'episodes' / item['job_id']
        path = folder / 'first_phase_result.json'
        require(sha256_file(path) == manifest['jobs'][item['job_id']]['files']['first_phase_result.json'], 'source phase changed')
        phase = read_json(path)['payload']
        require(phase['trace_sha256'] == old_plan['inputs'][(folder/'first_phase/trace.jsonl').relative_to(ROOT).as_posix()], 'phase trace identity')
        found = [r for r in available if r['item']['job_id'] == item['job_id']]
        inventory.append(dict(episode=item['job_id'], map_id=job['map_id'], solver_seed=item['solver_seed'],
            eligible_decisions=sorted(r['decision'] for r in found),
            repair_iterations=phase['summary']['repair_iterations'], source_stop=phase['stop_reason'],
            initial_conflicts=read_json(folder/'initial.json')['payload']['observation']['num_of_colliding_pairs']))
        files.append(path)
    write_json(out/'source_inventory.json',inventory)
    files.append(out/'source_inventory.json')
    files += [ROOT / f['file'] for f in frozen]
    for p in files:
        inputs[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    for p, digest in inputs.items():
        require(sha256_file(ROOT / p) == digest, 'changed during preparation: ' + p)
    plan = dict(config=cfg, roots=roots, coverage=coverage, inputs=inputs, frozen_folds=frozen,
                excluded_episodes=sorted(excluded), old_output=old_out.relative_to(ROOT).as_posix(),
                native_file=old_plan['native_file'], native_sha256=old_plan['native_sha256'], model=MODEL_PARAMS,
                admission=admission, commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    plan['binding'] = json_fingerprint(plan)
    write_json(out / 'plan.json', plan)
    if not admission:
        report = dict(binding=plan['binding'], decision='insufficient_independent_episode_coverage',
                      coverage=coverage, source_episodes=len(inventory),
                      eligible_source_episodes=sum(bool(r['eligible_decisions']) for r in inventory),
                      executed_new_branches=0, real_data_fits=0, collection_allowed=False, promotion_allowed=False)
        write_json(out/'admission_report.json',report)
        write_json(out/'run_status.json',dict(status='not_admitted',binding=plan['binding'],decision=report['decision']))
    return dict(admission=admission, binding=plan['binding'],
                proposed_budget=work_budget(roots,cfg), authorized_branch_jobs=work_budget(roots,cfg)['branch_jobs'] if admission else 0,
                coverage=coverage)


def verify():
    cfg = read_json(CONFIG)
    out = ROOT / cfg['output']
    plan = read_json(out / 'plan.json')
    require(plan['config'] == cfg and plan['model'] == MODEL_PARAMS and
            plan['binding'] == json_fingerprint({k:v for k,v in plan.items() if k != 'binding'}), 'plan changed')
    for p, digest in plan['inputs'].items():
        require(sha256_file(ROOT / p) == digest, 'registered input changed: ' + p)
    return plan, out


def collect(resume=False, limit=None):
    plan, out = verify()
    require_admission(plan)
    from scripts import run_sa_path_quality as q
    require(resume or not (out / 'roots').exists(), 'existing output requires resume')
    require(limit is None or 0 < limit <= len(plan['roots']), 'invalid limit')
    with q._CollectionRunLock(out, plan['binding'], 'state-coverage'):
        write_json(out / 'run_status.json', dict(status='running', binding=plan['binding']))
        try:
            for root in plan['roots'][:limit]:
                if (out / 'STOP_AFTER_ROOT').exists():
                    break
                job = dict(job_id=root['id'], target=root, plan=plan, parent_pid=os.getpid())
                result = q._run_jobs(pilot.root_worker, [job], workers=1, phase=root['id'],
                    output_root=out / 'progress' / root['id'], run_fingerprint=plan['binding'],
                    timeout_seconds=plan['config']['root_fuse_seconds'], stop_on_failure=True)
                require(len(result) == 1 and result[0]['status'] == 'ok', 'root failed; no retry/replacement')
                n = sum((out / 'roots' / r['id'] / 'receipt.json').exists() for r in plan['roots'])
                write_json(out / 'run_status.json', dict(status='running', binding=plan['binding'], completed_roots=n))
                print('ROOT', n, '/', len(plan['roots']), root['id'], flush=True)
            verify()
            complete = all((out / 'roots' / r['id'] / 'receipt.json').exists() for r in plan['roots'])
            write_json(out / 'run_status.json', dict(status='completed' if complete else 'paused', binding=plan['binding']))
        except BaseException as exc:
            write_json(out / 'run_status.json', dict(status='error', binding=plan['binding'], error=repr(exc)))
            raise
    return dict(complete=complete)


def index_root(job):
    plan, target = job
    cfg = plan['config']
    folder = ROOT / cfg['output'] / 'roots' / target['id']
    pilot.verify_receipt(folder, target, plan)
    root = read_json(folder / 'root.json')
    require(root['source'] == target and pilot.state_fingerprint(root['state']) == target['state_fingerprint'], 'root identity')
    require([c['candidate_id'] for c in root['candidates']] == target['selected'] and
            root['old_selected_id'] == target['anchor_id'], 'candidate/anchor identity')
    candidates = []
    for c, base in zip(root['candidates'], root['feature_rows'], strict=True):
        trials = []
        for t in range(cfg['trials']):
            row = read_json(folder / f"{c['candidate_id']}-t{t}.json")
            require((row['root_id'],row['candidate_id'],row['trial']) == (target['id'],c['candidate_id'],t), 'trial identity')
            label = None if row['status'] == 'censored' else pilot.branch_labels(
                row,root,dict(target=target,previous_best=root['previous_best']),cfg)
            trials.append(dict(trial=t, randomization_key=json_fingerprint([plan['binding'],target['id'],t]),
                stop=row['stop'], steps=len(row.get('events',[])), final_conflicts=row.get('final_conflicts',root['state']['num_of_colliding_pairs']),
                completed=None if label is None else bool(label['completion'])))
        candidates.append(dict(candidate_id=c['candidate_id'], agents=c['agents'],
                               features=profile_features(dict(base=base),'dynamic'), trials=trials))
    return dict(state_id=target['id'], map_id=target['map_id'], episode=target['item']['job_id'],
                decision=target['decision'], phase=target['phase'], anchor_id=target['anchor_id'],
                state_fingerprint=target['state_fingerprint'], history_fingerprint=target['history_fingerprint'],
                agent_ids=[a['id'] for a in root['state']['agents']], candidates=candidates)


def analyze():
    plan, out = verify()
    require_admission(plan)
    with ProcessPoolExecutor(max_workers=plan['config']['workers']) as pool:
        states = list(pool.map(index_root, [(plan,r) for r in plan['roots']]))
    names = sorted(states[0]['candidates'][0]['features'])
    data = dict(schema=SCHEMA, role='prospective_development', sampling='outcome_blind_same_state',
                source_kind='registered_prospective_collection', continuation_binding=plan['binding'],
                horizon=32, trial_count=8, feature_names=names, states=states)
    report = inspect_dataset(data)
    merge_data(read_json(ROOT / plan['old_output'] / 'training_index.json'), data, plan['binding'])
    report.update(binding=plan['binding'], data_fingerprint=json_fingerprint(data))
    verify()
    pilot.save_once(out / 'training_index.json', data)
    pilot.save_once(out / 'label_report.json', report)
    return {k:v for k,v in report.items() if k != 'states'}


def evaluate_fold(job):
    combined, new, frozen = job
    path = ROOT / frozen['file']
    require(sha256_file(path) == frozen['sha256'], 'frozen model changed')
    with path.open('rb') as stream:
        old_models = pickle.load(stream)
    require(old_models['train_ids'] == frozen['train_ids'] and old_models['train_maps'] == frozen['train_maps'], 'fold registration')
    augmented = fit_models(combined, frozen['map_id'])
    rows = []
    for s in new['states']:
        if s['map_id'] != frozen['map_id']:
            continue
        rows.append(dict(state_id=s['state_id'], old=predict(old_models,s), augmented=predict(augmented,s)))
    return dict(map_id=frozen['map_id'], old_train_ids=old_models['train_ids'],
                augmented_train_ids=augmented['train_ids'], predictions=rows)


def _fit_evaluate(plan,out):
    new = read_json(out / 'training_index.json')
    label_report = read_json(out / 'label_report.json')
    require(label_report['binding'] == plan['binding'] and label_report['data_fingerprint'] == json_fingerprint(new), 'new index changed')
    require(inspect_dataset(new) == {k:v for k,v in label_report.items() if k not in ('binding','data_fingerprint')}, 'label report changed')
    data_sha = sha256_file(out / 'training_index.json')
    receipt_path = out / 'evaluation_receipt.json'
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        require(receipt == dict(binding=plan['binding'], input_sha256=data_sha,
                                report_sha256=sha256_file(out/'model_report.json')), 'evaluation receipt changed')
        return dict(decision=read_json(out/'model_report.json')['decision'], resumed=True)
    require(not (out/'model_report.json').exists(), 'partial evaluation publication requires audit')
    combined = merge_data(read_json(ROOT / plan['old_output'] / 'training_index.json'),new,plan['binding'])
    if not label_report['training_data_contract_passed']:
        report = dict(binding=plan['binding'], decision='insufficient_labels_no_augmented_training',
                      reasons=label_report['rejection_reasons'], promotion_allowed=False)
    else:
        # Two fixed designs, not a search over model or gate hyperparameters.
        with ProcessPoolExecutor(max_workers=min(plan['config']['workers'],len(plan['frozen_folds']))) as pool:
            folds=[]
            for fold in pool.map(evaluate_fold, [(combined,new,f) for f in plan['frozen_folds']]):
                folds.append(fold)
                print('EVALUATED_MAP',len(folds),'/',len(plan['frozen_folds']),flush=True)
        lookup={p['state_id']:p for f in folds for p in f['predictions']}
        rows=[]
        for s in new['states']:
            values={c['candidate_id']:[int(t['completed']) for t in sorted(c['trials'],key=lambda t:t['trial'])] for c in s['candidates']}
            diagnostic=split_diagnostics(values,s['anchor_id'])
            rates=diagnostic['rates']
            prediction=lookup[s['state_id']]
            result=dict(frozen=rates[s['anchor_id']],uniform=sum(rates.values())/len(rates),oracle=max(rates.values()),
                        cross_half=diagnostic['cross_half_rate'])
            sizes={c['candidate_id']:len(c['agents']) for c in s['candidates']}
            selected_sizes={}
            for dataset in ('old','augmented'):
                for model in ('paired','direct'):
                    cid=prediction[dataset][model]['selected']
                    result[dataset+'_'+model]=rates[cid]
                    selected_sizes[dataset+'_'+model]=sizes[cid]
            rows.append(dict(state_id=s['state_id'],map_id=s['map_id'],phase=s['phase'],rates=result,
                             sizes=selected_sizes,split=diagnostic,prediction=prediction))
        summary={k:sum(r['rates'][k] for r in rows)/len(rows) for k in rows[0]['rates']}
        comparisons,signals={},{}
        for model in ('paired','direct'):
            method='augmented_'+model
            comparisons[model]={b:contrast(rows,method,b,plan['config']['seed'],plan['config']['bootstrap']) for b in ('frozen','uniform','old_'+model)}
            comparisons[model]['old_same_model']=comparisons[model].pop('old_'+model)
            signals[model]=signal_gate(comparisons[model])
        active=[r for r in rows if r['split']['informative']]
        report=dict(binding=plan['binding'],input_sha256=data_sha,folds=folds,states=rows,summary=summary,
                    comparisons=comparisons,limited_signals=signals,
                    informative_states=len(active),informative_maps=len({r['map_id'] for r in active}),
                    informative_jaccard=sum(r['split']['mean_jaccard'] for r in active)/len(active) if active else None,
                    decision='limited_state_coverage_signal' if any(signals.values()) else 'stop_h32_tabular_expansion',
                    no_ttf=True,promotion_allowed=False,runtime_integration_allowed=False,independent_confirmation=False)
    verify()
    require(sha256_file(out/'training_index.json') == data_sha, 'input changed during training')
    pilot.save_once(out/'model_report.json',report)
    write_json(receipt_path,dict(binding=plan['binding'],input_sha256=data_sha,report_sha256=sha256_file(out/'model_report.json')))
    return {k:report[k] for k in ('decision','summary','comparisons','limited_signals','informative_states','informative_maps','informative_jaccard') if k in report}


def fit_evaluate():
    plan,out=verify()
    require_admission(plan)
    from scripts import run_sa_path_quality as q
    with q._CollectionRunLock(out,plan['binding'],'state-coverage-training'):
        write_json(out/'training_status.json',dict(status='running',binding=plan['binding']))
        try:
            result=_fit_evaluate(plan,out)
            write_json(out/'training_status.json',dict(status='completed',binding=plan['binding'],decision=result['decision']))
            return result
        except BaseException as exc:
            write_json(out/'training_status.json',dict(status='error',binding=plan['binding'],error=repr(exc)))
            raise


def main():
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','dry-run','collect','analyze','fit-evaluate','request-stop','verify'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.phase=='prepare': result=prepare(args.resume)
    elif args.phase=='collect': result=collect(args.resume,args.limit)
    elif args.phase=='analyze': result=analyze()
    elif args.phase=='fit-evaluate': result=fit_evaluate()
    else:
        plan,out=verify()
        if args.phase=='request-stop':
            write_json(out/'STOP_AFTER_ROOT',dict(binding=plan['binding'],requested=True))
            result=dict(stop_after_current_root=True)
        else:
            result=dict(binding=plan['binding'],admission=plan['admission'],
                        proposed_budget=work_budget(plan['roots'],plan['config']),
                        authorized_branch_jobs=work_budget(plan['roots'],plan['config'])['branch_jobs'] if plan['admission'] else 0)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
