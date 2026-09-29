"""One on-policy completion update from frozen A, with map-disjoint development comparison."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_terminal_efficiency as old
from experiments.sa_raw_selection_fast import _bind

run, raw, require = old.run, old.raw, old.require
CONFIG = 'configs/sa_completion_continuation.json'
ARMS = ('parent', 'completion')
# Private bindings reuse the audited update without changing the historical module.
objective = SimpleNamespace(**(vars(old.objective) | {'OBJECTIVES': ('completion',)}))


def configuration():
    cfg = run.read_json(ROOT/CONFIG)
    fixed = dict(workers=20, max_decisions=None, node_budget=25000000,
        pp_safety_seconds=20., episode_safety_seconds=900., process_fuse_seconds=960.,
        maximum_updates_per_arm=1, train_replicas=4, comparison_replicas=2,
        objective='completion', bootstrap=5000, bootstrap_seed=2026093001,
        formal_ttf=False, automatic_promotion=False,
        phase='completion-continuation-train-20260930',
        comparison_phase='completion-continuation-eval-20260930',
        parent_sha256='81727429f423734aef544c1e12ad61fae965a2118898fadafeab699dd9307062',
        source_registration_sha256='732f163e2bd4c6506cd8df5fa13390b28ac3dd626bd8a031c750a69dd61d15f4')
    require(all(cfg[k] == v for k,v in fixed.items()), 'fixed continuation scope changed')
    return cfg


def conditions(jobs, split):
    rows, initial = {}, {}
    for j in jobs:
        require(j['split'] == split, 'source split changed')
        c = {k:j[k] for k in ('case','pair_id','solver_seed','split')}
        key = j['pair_id']
        require(rows.setdefault(key,c) == c, 'inconsistent source condition')
        require(initial.setdefault(key,j['expected_initial']) == j['expected_initial'], 'initial mismatch')
    require(len(rows) == 24 and len({c['case']['map_id'] for c in rows.values()}) == 6,
            'expected 24 conditions on six maps')
    return [rows[k] for k in sorted(rows)], initial


def validate_separation(train, evaluation, inputs):
    for field in ('map_id','task_id'):
        require(not {c['case'][field] for c in train} & {c['case'][field] for c in evaluation},
                'train/evaluation identity leak')
    for field in ('map_file','scenario_file'):
        a = {inputs[c['case']['files'][field]] for c in train}
        b = {inputs[c['case']['files'][field]] for c in evaluation}
        require(not a & b, 'train/evaluation content leak')


def prepare():
    cfg = configuration()
    out, src = ROOT/cfg['output'], ROOT/cfg['source']
    require(not out.exists(), 'existing output; verify/resume instead of overwrite')
    require(run.sha256_file(src/'registration.json') == cfg['source_registration_sha256'], 'source registration SHA')
    source = run.check_seal(run.read_json(src/'registration.json'))
    update = run.check_seal(run.read_json(src/'update.completion.json'))
    parent = run.read_json(ROOT/cfg['parent_model'])
    require(update['binding'] == source['binding'] and update['objective'] == 'completion' and
            update['updated'] and parent['iteration'] == 2, 'frozen A lineage')
    require(run.sha256_file(ROOT/cfg['parent_model']) == cfg['parent_sha256'] == update['model_sha256'] and
            raw.validate_bundle(parent) == update['policy_sha256'], 'frozen A SHA')
    train, initial = conditions(source['train_jobs'], 'train')
    evaluate, eval_initial = conditions(source['comparison_jobs'], 'development_holdout')
    validate_separation(train,evaluate,source['inputs'])
    phases = {j['phase'] for j in source['train_jobs']+source['comparison_jobs']}
    require(not phases & {cfg['phase'],cfg['comparison_phase']}, 'reused random streams')
    inputs = {}
    def register(path, expected=None):
        path = Path(path)
        digest = run.sha256_file(path)
        require(expected is None or digest == expected, 'changed source: '+str(path))
        inputs[path.relative_to(ROOT).as_posix()] = digest
    for c in train+evaluate:
        for name in c['case']['files'].values():
            register(ROOT/name,source['inputs'][name])
    plans = deepcopy(source['plans'])
    for plan in plans.values():
        require(plan['template']['environment']['max_repair_iterations'] == 0, 'native decision cap')
        register(ROOT/plan['native_file'],plan['config']['native_sha256'])
    for path in (ROOT/CONFIG, ROOT/cfg['parent_model'],src/'registration.json',src/'update.completion.json',
                 ROOT/'docs/SA_COMPLETION_CONTINUATION_PROTOCOL_ZH.md',
                 ROOT/'tests/evaluation/test_sa_completion_continuation.py'):
        register(path)
    for directory in ('scripts','experiments','lns2_selector'):
        for path in (ROOT/directory).rglob('*.py'):
            register(path)
    for path in (ROOT/'artifacts/initlns-closed-loop-controller-v2').glob('*.json'):
        register(path)
    body = dict(schema=cfg['schema'],config=cfg,inputs=inputs,plans=plans,parent=parent,
        train_jobs=old.schedule(train,initial,cfg['phase'],4,['parent']),
        comparison_jobs=old.schedule(evaluate,eval_initial,cfg['comparison_phase'],2,ARMS),
        source_commit=run.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        no_ttf=True,no_promotion=True,evaluation_role='previously_viewed_map_disjoint_development',
        timing_results_used_for_training=False)
    body['binding'] = run.json_fingerprint(body)
    with old.recovery.strict_lock(out,body['binding'],'completion-continuation-prepare'):
        run.once(out/'registration.json',run.sealed(body))
    return dict(prepared=True,train=96,comparison=96,workers=20,max_decisions=None,
                no_ttf=True,binding=body['binding'])


def _call(function, *args):
    return _bind(function,ROOT=ROOT,configuration=configuration,verify=verify,model_spec=model_spec,
        jobs_for=jobs_for,audited_rows=audited_rows,objective=objective,ARMS=ARMS)(*args)


def verify():
    return _call(old.verify)


def model_spec(reg, arm):
    return _call(old.model_spec,reg,arm)


def jobs_for(reg,lane):
    require(lane in ('train','comparison'), 'unknown lane')
    jobs = _call(old.jobs_for,reg,lane)
    for job in jobs:
        job['iteration'] = reg['parent']['iteration'] + int(job['comparison_arm'] != 'parent')
    return jobs


def audited_rows(reg,out,lane):
    return _call(old.audited_rows,reg,out,lane)


def collect(lane,resume=False):
    return _call(old.collect,lane,resume)


def audit(lane):
    return _call(old.audit,lane)


def train():
    return _call(old.train)


def parity(platform):
    return _call(old.parity,platform)


def signal(c):
    work = c['metrics']['generated']['change_percent']
    return c['challenger_success'] >= c['baseline_success'] and (
        c['challenger_success'] > c['baseline_success'] or (work is not None and work <= -5.))


def report():
    reg,out = verify()
    update = run.check_seal(run.read_json(out/'update.completion.json'))
    require(update['binding'] == reg['binding'], 'update binding')
    result = dict(binding=reg['binding'],no_ttf=True,no_promotion=True,independent_generalization=False,
        update_sha256=run.sha256_file(out/'update.completion.json'),updated=update['updated'])
    if not update['updated']:
        result.update(decision='no_admissible_update_keep_A',development_signal=False)
    else:
        rows = [dict(r,comparison_arm=j['comparison_arm']) for j,r in audited_rows(reg,out,'comparison')]
        cfg = reg['config']
        c = objective.contrast(rows,*ARMS,bootstrap=cfg['bootstrap'],seed=cfg['bootstrap_seed'])
        variants = {j['pair_id']:j['case']['task_variant'] for j in reg['comparison_jobs']}
        result.update(episodes=rows,contrast=c,development_signal=signal(c),
            by_density={v:objective.contrast([r for r in rows if variants[r['pair_id']]==v],*ARMS,
                bootstrap=cfg['bootstrap'],seed=cfg['bootstrap_seed']) for v in sorted(set(variants.values()))},
            decision='development_signal_not_ttf' if signal(c) else 'no_development_gain_keep_A',
            comparison_audit_sha256=run.sha256_file(out/'comparison.audit.json'))
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
