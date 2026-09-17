"""Frozen 31-state H32 collection; separate from the rejected equal-quota plan."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments._common import read_json,write_json,sha256_file,json_fingerprint
from experiments.sa_paired_completion import SCHEMA,MODEL_PARAMS,require,inspect_dataset
from experiments.sa_paired_sampling import work_budget
from experiments.sa_state_coverage import merge_data,predict,contrast,signal_gate
from experiments.sa_unbalanced_coverage import fit_models,summarize,validate_cohort
from experiments.sa_completion_contract import split_diagnostics
from scripts import run_sa_paired_completion_pilot as pilot
from scripts.run_sa_state_coverage import index_root

CONFIG=ROOT/'configs/sa_unbalanced_coverage.json'


def freeze_fold(job):
    data,held,folder=job
    models=fit_models(data,held)
    path=Path(folder)/(held+'.pkl')
    path.parent.mkdir(parents=True,exist_ok=True)
    require(not path.exists(),'frozen fold already exists')
    with path.open('xb') as stream:
        pickle.dump(models,stream,protocol=4)
    return dict(map_id=held,file=path.relative_to(ROOT).as_posix(),sha256=sha256_file(path),
                train_ids=models['train_ids'],train_maps=models['train_maps'])


def prepare():
    from scripts.run_sa_state_coverage import verify as verify_previous
    settings=read_json(CONFIG)
    previous,previous_out=verify_previous()
    require(sha256_file(ROOT/settings['previous_plan'])==settings['previous_plan_sha256'],'old failed plan changed')
    require(not previous['admission'] and not (previous_out/'roots').exists() and not previous['frozen_folds'],
            'revision must precede any new labels or training')
    require(json_fingerprint(previous['roots'])==settings['root_cohort_sha256'],'root cohort changed')
    validate_cohort(previous['roots'],settings['map_counts'],previous['excluded_episodes'])
    require(len(previous['roots'])==settings['states']==31 and len(settings['map_counts'])==8,'registered cohort size changed')
    cfg=dict(previous['config'],output=settings['output'],states=settings['states'],schema=settings['schema'])
    out=ROOT/cfg['output']
    require(not out.exists(),'output already exists; do not overwrite preparation')
    old_path=ROOT/previous['old_output']/'training_index.json'
    old=read_json(old_path)
    require(inspect_dataset(old)['training_data_contract_passed'],'old data contract failed')
    inputs=dict(previous['inputs'])
    files=[CONFIG,Path(__file__),ROOT/'experiments/sa_unbalanced_coverage.py',
           ROOT/'tests/evaluation/test_sa_unbalanced_coverage.py',ROOT/'docs/SA_UNBALANCED_COVERAGE_PROTOCOL_ZH.md',
           ROOT/settings['previous_plan'],previous_out/'admission_report.json']
    for p in files:
        inputs[p.relative_to(ROOT).as_posix()]=sha256_file(p)
    for p,digest in inputs.items():
        require(sha256_file(ROOT/p)==digest,'changed input: '+p)
    with ProcessPoolExecutor(max_workers=min(cfg['workers'],len(settings['map_counts']))) as pool:
        frozen=list(pool.map(freeze_fold,[(old,m,str(out/'old_fold_models')) for m in sorted(settings['map_counts'])]))
    for f in frozen:
        inputs[f['file']]=f['sha256']
    plan=dict(config=cfg,settings=settings,roots=previous['roots'],inputs=inputs,frozen_folds=frozen,
              old_output=previous['old_output'],excluded_episodes=previous['excluded_episodes'],model=MODEL_PARAMS,
              native_file=previous['native_file'],native_sha256=previous['native_sha256'],admission=True,
              commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    plan['binding']=json_fingerprint(plan)
    write_json(out/'plan.json',plan)
    write_json(out/'run_status.json',dict(status='prepared',binding=plan['binding'],completed_roots=0))
    verify()
    return dict(binding=plan['binding'],budget=work_budget(plan['roots'],cfg),frozen_old_data_fits=2*len(frozen))


def verify():
    settings=read_json(CONFIG)
    out=ROOT/settings['output']
    plan=read_json(out/'plan.json')
    require(plan['settings']==settings and plan['model']==MODEL_PARAMS and
            plan['binding']==json_fingerprint({k:v for k,v in plan.items() if k!='binding'}),'plan identity changed')
    require(json_fingerprint(plan['roots'])==settings['root_cohort_sha256'],'root cohort changed')
    validate_cohort(plan['roots'],settings['map_counts'],plan['excluded_episodes'])
    for name,digest in plan['inputs'].items():
        require(sha256_file(ROOT/name)==digest,'registered input changed: '+name)
    return plan,out


def collect(resume=False,limit=None):
    plan,out=verify()
    from scripts import run_sa_path_quality as q
    require(resume or not (out/'roots').exists(),'existing collection requires resume')
    require(limit is None or 0<limit<=len(plan['roots']),'invalid root limit')
    with q._CollectionRunLock(out,plan['binding'],'unbalanced-coverage'):
        write_json(out/'run_status.json',dict(status='running',binding=plan['binding']))
        try:
            for root in plan['roots'][:limit]:
                if (out/'STOP_AFTER_ROOT').exists():
                    break
                job=dict(job_id=root['id'],target=root,plan=plan,parent_pid=os.getpid())
                results=q._run_jobs(pilot.root_worker,[job],workers=1,phase=root['id'],
                    output_root=out/'progress'/root['id'],run_fingerprint=plan['binding'],
                    timeout_seconds=plan['config']['root_fuse_seconds'],stop_on_failure=True)
                require(len(results)==1 and results[0]['status']=='ok','root failed; audit before retry')
                n=sum((out/'roots'/r['id']/'receipt.json').exists() for r in plan['roots'])
                write_json(out/'run_status.json',dict(status='running',binding=plan['binding'],completed_roots=n))
                print('ROOT',n,'/',len(plan['roots']),root['id'],flush=True)
            verify()
            complete=all((out/'roots'/r['id']/'receipt.json').exists() for r in plan['roots'])
            write_json(out/'run_status.json',dict(status='completed' if complete else 'paused',binding=plan['binding']))
        except BaseException as exc:
            write_json(out/'run_status.json',dict(status='error',binding=plan['binding'],error=repr(exc)))
            raise
    return dict(complete=complete)


def analyze():
    plan,out=verify()
    with ProcessPoolExecutor(max_workers=plan['config']['workers']) as pool:
        states=list(pool.map(index_root,[(plan,r) for r in plan['roots']]))
    data=dict(schema=SCHEMA,role='prospective_development',sampling='outcome_blind_same_state',
              source_kind='registered_prospective_collection',continuation_binding=plan['binding'],
              horizon=32,trial_count=8,feature_names=sorted(states[0]['candidates'][0]['features']),states=states)
    report=inspect_dataset(data)
    merge_data(read_json(ROOT/plan['old_output']/'training_index.json'),data,plan['binding'])
    report.update(binding=plan['binding'],data_fingerprint=json_fingerprint(data))
    verify()
    pilot.save_once(out/'training_index.json',data)
    pilot.save_once(out/'label_report.json',report)
    return {k:v for k,v in report.items() if k!='states'}


def evaluate_fold(job):
    combined,new,frozen=job
    path=ROOT/frozen['file']
    require(sha256_file(path)==frozen['sha256'],'old model changed')
    with path.open('rb') as stream:
        old=pickle.load(stream)
    require(old['train_ids']==frozen['train_ids'] and old['train_maps']==frozen['train_maps'],'old fold identity')
    augmented=fit_models(combined,frozen['map_id'])
    rows=[dict(state_id=s['state_id'],old=predict(old,s),augmented=predict(augmented,s))
          for s in new['states'] if s['map_id']==frozen['map_id']]
    return dict(map_id=frozen['map_id'],old_train_ids=old['train_ids'],augmented_train_ids=augmented['train_ids'],
                augmented_state_weights=augmented['state_weights'],predictions=rows)


def make_report(plan,new,folds):
    predictions={p['state_id']:p for f in folds for p in f['predictions']}
    require(set(predictions)=={s['state_id'] for s in new['states']},'prediction coverage mismatch')
    rows=[]
    for s in new['states']:
        values={c['candidate_id']:[int(t['completed']) for t in sorted(c['trials'],key=lambda t:t['trial'])] for c in s['candidates']}
        split=split_diagnostics(values,s['anchor_id'])
        rates=split['rates']
        prediction=predictions[s['state_id']]
        scores=dict(frozen=rates[s['anchor_id']],uniform=sum(rates.values())/len(rates),oracle=max(rates.values()),
                    cross_half=split['cross_half_rate'])
        sizes={c['candidate_id']:len(c['agents']) for c in s['candidates']}
        chosen_sizes={}
        for version in ('old','augmented'):
            for model in ('paired','direct'):
                cid=prediction[version][model]['selected']
                scores[version+'_'+model]=rates[cid]
                chosen_sizes[version+'_'+model]=sizes[cid]
        rows.append(dict(state_id=s['state_id'],map_id=s['map_id'],phase=s['phase'],rates=scores,
                         sizes=chosen_sizes,split=split,prediction=prediction))
    comparisons,signals={},{}
    cfg=plan['config']
    for model in ('paired','direct'):
        comparisons[model]={b:contrast(rows,'augmented_'+model,b,cfg['seed'],cfg['bootstrap'])
                            for b in ('frozen','uniform','old_'+model)}
        comparisons[model]['old_same_model']=comparisons[model].pop('old_'+model)
        signals[model]=signal_gate(comparisons[model])
    active=[r for r in rows if r['split']['informative']]
    return dict(binding=plan['binding'],states=rows,folds=folds,summary=summarize(rows),comparisons=comparisons,
        limited_signals=signals,informative_states=len(active),informative_maps=len({r['map_id'] for r in active}),
        informative_jaccard=sum(r['split']['mean_jaccard'] for r in active)/len(active) if active else None,
        decision='limited_state_coverage_signal' if any(signals.values()) else 'stop_h32_tabular_expansion',
        promotion_allowed=False,runtime_integration_allowed=False,no_ttf=True,independent_confirmation=False)


def _fit_evaluate(plan,out):
    new=read_json(out/'training_index.json')
    label=read_json(out/'label_report.json')
    require(label['binding']==plan['binding'] and label['data_fingerprint']==json_fingerprint(new),'new data identity')
    require(inspect_dataset(new)=={k:v for k,v in label.items() if k not in ('binding','data_fingerprint')},'label report identity')
    digest=sha256_file(out/'training_index.json')
    receipt=out/'evaluation_receipt.json'
    if receipt.exists():
        require(read_json(receipt)==dict(binding=plan['binding'],input_sha256=digest,
            report_sha256=sha256_file(out/'model_report.json')),'evaluation receipt changed')
        return dict(decision=read_json(out/'model_report.json')['decision'],resumed=True)
    require(not (out/'model_report.json').exists(),'partial evaluation needs audit')
    combined=merge_data(read_json(ROOT/plan['old_output']/'training_index.json'),new,plan['binding'])
    if not label['training_data_contract_passed']:
        report=dict(binding=plan['binding'],decision='insufficient_labels_no_augmented_training',
                    reasons=label['rejection_reasons'],promotion_allowed=False)
    else:
        with ProcessPoolExecutor(max_workers=min(plan['config']['workers'],len(plan['frozen_folds']))) as pool:
            folds=[]
            for f in pool.map(evaluate_fold,[(combined,new,f) for f in plan['frozen_folds']]):
                folds.append(f)
                print('EVALUATED_MAP',len(folds),'/',len(plan['frozen_folds']),flush=True)
        report=make_report(plan,new,folds)
    verify()
    require(sha256_file(out/'training_index.json')==digest,'input changed during fitting')
    pilot.save_once(out/'model_report.json',report)
    write_json(receipt,dict(binding=plan['binding'],input_sha256=digest,report_sha256=sha256_file(out/'model_report.json')))
    return {k:report[k] for k in ('decision','summary','comparisons','limited_signals','informative_states','informative_maps','informative_jaccard') if k in report}


def fit_evaluate():
    plan,out=verify()
    from scripts import run_sa_path_quality as q
    with q._CollectionRunLock(out,plan['binding'],'unbalanced-training'):
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
    parser.add_argument('phase',choices=('prepare','dry-run','collect','analyze','fit-evaluate','verify','request-stop'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.phase=='prepare': result=prepare()
    elif args.phase=='collect': result=collect(args.resume,args.limit)
    elif args.phase=='analyze': result=analyze()
    elif args.phase=='fit-evaluate': result=fit_evaluate()
    else:
        plan,out=verify()
        if args.phase=='request-stop':
            write_json(out/'STOP_AFTER_ROOT',dict(binding=plan['binding'],requested=True))
            result=dict(stop_after_current_root=True)
        else:
            result=dict(binding=plan['binding'],admission=plan['admission'],budget=work_budget(plan['roots'],plan['config']))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
