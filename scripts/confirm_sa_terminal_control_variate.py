"""Frozen prior-batch baselines on fresh same-A trajectories; no parameter update or formal TTF."""
import argparse
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
from experiments import sa_terminal_control_variate_confirmation as statistics
from experiments.sa_raw_selection_fast import _bind
from scripts import run_sa_completion_replicability as source
from scripts import audit_sa_terminal_control_variate as baseline_source

run, require = source.run, source.require
CONFIG = 'configs/sa_terminal_control_variate_confirmation.json'


def configuration():
    cfg = run.read_json(ROOT/CONFIG)
    fixed = dict(schema='lns2.sa_terminal_control_variate_confirmation.v1',
        output='build/sa-terminal-control-variate-confirmation-v1', source='build/sa-completion-credit-replicability-v1',
        source_report_sha256='18466f108adfbf051bed14059c94074458510643a6ed6c83bff0fa8627fff923',
        baseline_source='build/sa-terminal-control-variate-v1',
        baseline_report_sha256='6669c6eb8941f5b88011525c316449d9a62eda1ea647d3e90a2a4956c54c2bb3',
        parent_model='build/sa-terminal-efficiency-v1/models/completion.json',
        parent_sha256='81727429f423734aef544c1e12ad61fae965a2118898fadafeab699dd9307062',
        phase='terminal-control-variate-confirmation-20261001', replicas=16, workers=20, max_decisions=None,
        node_budget=25000000, pp_safety_seconds=20., episode_safety_seconds=900., process_fuse_seconds=960.,
        bootstrap=2000, bootstrap_seed=2026100102, minimum_second_moment_reduction=.1,
        maximum_condition_increase=.1, maximum_updates_per_arm=0, formal_ttf=False, automatic_promotion=False)
    require(all(cfg[k] == v for k, v in fixed.items()), 'fixed confirmation scope changed')
    return cfg


def prepare():
    cfg = configuration()
    out = ROOT/cfg['output']
    require(not out.exists(), 'existing output; verify/resume instead')
    proof = baseline_source.verify_report()
    require(proof['report_sha256'] == cfg['baseline_report_sha256'], 'prior baseline evidence changed')
    require(source.verify_report()['report_sha256'] == cfg['source_report_sha256'], 'prior trajectory evidence changed')
    old, src = source.verify()
    base_reg, base_out = baseline_source.verify()
    require(src == ROOT/cfg['source'] and base_out == ROOT/cfg['baseline_source'] and
            old['config']['parent_sha256'] == cfg['parent_sha256'], 'source lineage')
    fit = run.check_seal(run.read_json(src/'scores.json'))
    require(fit['binding'] == old['binding'] and len(fit['rows']) == 64, 'fit score coverage')
    original, _ = source.old_statistics(old)
    known = original + fit['rows']
    require(len({r['rng_stream_id'] for r in known}) == 80, 'historical streams')
    require(cfg['phase'] not in {j['phase'] for j in old['train_jobs'] + old['original_jobs']}, 'phase reused')
    frozen = {}
    for c in old['conditions']:
        group = sorted([r for r in fit['rows'] if r['pair_id'] == c['pair_id']], key=lambda r: r['replica'])
        frozen[c['pair_id']] = statistics.freeze_baselines(group)
        require(frozen[c['pair_id']]['initial_fingerprint'] == c['expected_initial'], 'fit initial identity')
    inputs = dict(base_reg['inputs'])
    for p in (src/'scores.json', src/'report.json', base_out/'registration.json', base_out/'report.json',
              ROOT/CONFIG, Path(__file__), Path(statistics.__file__),
              ROOT/'tests/evaluation/test_sa_terminal_control_variate_confirmation.py',
              ROOT/'docs/SA_TERMINAL_CONTROL_VARIATE_CONFIRMATION_PROTOCOL_ZH.md'):
        inputs[p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
    plan = deepcopy(old['plans']['train'])
    require(plan['template']['environment']['max_repair_iterations'] == 0, 'hidden iteration cap')
    body = dict(schema=cfg['schema'], config=cfg, inputs=inputs, parent=old['parent'], plans={'train': plan},
        conditions=old['conditions'], train_jobs=source.schedule(old['conditions'], cfg['phase'], 16),
        baselines=frozen, known_streams=sorted(r['rng_stream_id'] for r in known),
        known_episode_ids=sorted(r['episode_id'] for r in known), source_binding=old['binding'],
        baseline_binding=base_reg['binding'], source_commit=run.subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        no_training=True, no_ttf=True, no_promotion=True, no_refit=True)
    body['binding'] = run.json_fingerprint(body)
    with source.prior.old.recovery.strict_lock(out, body['binding'], 'baseline-confirmation-prepare'):
        run.once(out/'registration.json', run.sealed(body))
    return dry_run_body(body)


def verify():
    reg, out = _bind(source.prior.old.verify, ROOT=ROOT, configuration=configuration)()
    fit = run.check_seal(run.read_json(ROOT/reg['config']['source']/'scores.json'))
    require(fit['binding'] == reg['source_binding'] and len(fit['rows']) == 64, 'fit lineage changed')
    expected = {pair: statistics.freeze_baselines(sorted(
        [r for r in fit['rows'] if r['pair_id'] == pair], key=lambda r: r['replica'])) for pair in reg['baselines']}
    statistics.compare_derived(reg['baselines'], expected)
    require(reg['no_training'] and reg['no_ttf'] and reg['no_refit'] and reg['no_promotion'], 'scope')
    return reg, out


def model_spec(reg, arm):
    return source.model_spec(reg, arm)


def jobs_for(reg, lane='train'):
    require(lane == 'train', 'no heldout lane')
    jobs = _bind(source.prior.old.jobs_for, ROOT=ROOT, configuration=configuration, model_spec=model_spec)(reg, lane)
    for j in jobs:
        j['iteration'] = reg['parent']['iteration']
    return jobs


def call(function, *args):
    return _bind(function, ROOT=ROOT, configuration=configuration, verify=verify,
                 model_spec=model_spec, jobs_for=jobs_for, audited_rows=audited_rows)(*args)


def audited_rows(reg, out, lane='train'):
    return call(source.prior.old.audited_rows, reg, out, lane)


def collect(resume=False):
    return call(source.prior.old.collect, 'train', resume)


def audit():
    return call(source.prior.old.audit, 'train')


def dry_run_body(reg):
    require(len(reg['train_jobs']) == 64 and len(reg['baselines']) == 4, 'fixed job inventory')
    return dict(binding=reg['binding'], episodes=64, conditions=4, replicas=16, workers=20,
        max_decisions=None, node_budget_per_episode=25000000, total_node_boundary=1600000000,
        atomic_repair_may_overshoot=True, process_fuse_batch_upper_seconds=3840.,
        runtime_estimate_note='Prior collection 1043s plus native audit 654s; estimate 30-45 minutes, not TTF.',
        maximum_parameter_updates=0, formal_ttf=False, no_refit=True,
        baseline_values={p: b['baselines'] for p, b in reg['baselines'].items()})


def summarize(reg, rows):
    require(len(rows) == 64 and len({r['episode_id'] for r in rows}) == 64 and
            len({r['rng_stream_id'] for r in rows}) == 64 and
            not set(reg['known_episode_ids']) & {r['episode_id'] for r in rows} and
            {r['job_id'] for r in rows} == {j['job_id'] for j in reg['train_jobs']}, 'fresh inventory')
    conditions = {}
    for pair, frozen in reg['baselines'].items():
        group = sorted([r for r in rows if r['pair_id'] == pair], key=lambda r: r['replica'])
        conditions[pair] = statistics.condition_report(group, frozen, reg['known_streams'])
    cfg = reg['config']
    result = statistics.summarize(conditions, cfg['minimum_second_moment_reduction'],
        cfg['maximum_condition_increase'], cfg['bootstrap'], cfg['bootstrap_seed'])
    result.update(schema=cfg['schema']+'.report', binding=reg['binding'],
                  fit_report_sha256=cfg['source_report_sha256'], prior_diagnostic_sha256=cfg['baseline_report_sha256'])
    return result


def analyze():
    reg, out = verify()
    jobs = [j for j, _ in audited_rows(reg, out)]
    require(len(jobs) == 64, 'full native audit required')
    with source.prior.old.recovery.strict_lock(out, reg['binding'], 'baseline-confirmation-analysis'):
        require(not (out/'report.json').exists(), 'existing report; verify instead')
        run.write_json(out/'run_status.json', dict(status='running', phase='score', binding=reg['binding']))
        try:
            if (out/'scores.json').exists():
                doc = run.check_seal(run.read_json(out/'scores.json'))
                require(doc['binding'] == reg['binding'], 'stale resumed scores')
                rows = doc['rows']
            else:
                rows = []
                with ProcessPoolExecutor(max_workers=20) as pool:
                    for r in pool.map(source.score_worker, jobs):
                        rows.append(r)
                        if len(rows) % 8 == 0:
                            print(json.dumps(dict(phase='score_replay', done=len(rows), total=64)), flush=True)
                summarize(reg, rows)
                run.once(out/'scores.json', run.sealed(dict(binding=reg['binding'], rows=rows)))
            result = summarize(reg, rows)
            result.update(audit_sha256=run.sha256_file(out/'train.audit.json'),
                          complete_sha256=run.sha256_file(out/'train.complete.json'),
                          scores_sha256=run.sha256_file(out/'scores.json'))
            verify()
            run.once(out/'report.json', run.sealed(result))
            run.write_json(out/'run_status.json', dict(status='complete', binding=reg['binding'], decision=result['decision']))
        except Exception as error:
            run.write_json(out/'run_status.json', dict(status='failed', binding=reg['binding'],
                error=type(error).__name__, message=str(error)))
            raise
    return dict(complete=True, decision=result['decision'], report_sha256=run.sha256_file(out/'report.json'),
                successes=result['successes'], second_moment=result['condition_equal_second_moment'])


def verify_report():
    reg, out = verify()
    result = run.check_seal(run.read_json(out/'report.json'))
    for field, path in (('audit_sha256', 'train.audit.json'), ('complete_sha256', 'train.complete.json'),
                        ('scores_sha256', 'scores.json')):
        require(result[field] == run.sha256_file(out/path), 'changed completed evidence: '+path)
    scores = run.check_seal(run.read_json(out/'scores.json'))
    require(scores['binding'] == reg['binding'] and len(list(audited_rows(reg, out))) == 64, 'score/audit lineage')
    expected = summarize(reg, scores['rows'])
    statistics.compare_derived({k: result[k] for k in expected}, expected)
    return dict(verified=True, report_sha256=run.sha256_file(out/'report.json'), decision=result['decision'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare', 'dry-run', 'collect', 'audit', 'analyze', 'verify', 'stop'))
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.phase == 'collect':
        result = collect(args.resume)
    elif args.phase == 'dry-run':
        result = dry_run_body(verify()[0])
    elif args.phase == 'verify':
        result = verify_report() if (ROOT/configuration()['output']/'report.json').exists() else dict(verified=True, binding=verify()[0]['binding'])
    elif args.phase == 'stop':
        reg, out = verify()
        run.write_json(out/'STOP_AFTER_BATCH', dict(requested=True))
        result = dict(stop_after_current_batch=True)
    else:
        result = globals()[args.phase]()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
