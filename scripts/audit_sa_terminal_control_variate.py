"""prepare/analyze/verify a terminal-score baseline diagnostic, without training or solver calls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import sa_terminal_control_variate as statistics
from scripts import run_sa_completion_replicability as source

run, require = source.run, source.require
CONFIG = 'configs/sa_terminal_control_variate.json'


def configuration():
    cfg = run.read_json(ROOT / CONFIG)
    fixed = dict(schema='lns2.sa_terminal_control_variate.v1', output='build/sa-terminal-control-variate-v1',
        source='build/sa-completion-credit-replicability-v1',
        source_report_sha256='18466f108adfbf051bed14059c94074458510643a6ed6c83bff0fa8627fff923',
        workers=20, methods=['mean', 'score_squared'],
        partitions='all_balanced_complement_deduplicated', minimum_second_moment_reduction=.1,
        maximum_condition_increase=.1, maximum_parameter_updates=0, formal_ttf=False, new_solver_calls=0)
    require(all(cfg[k] == v for k, v in fixed.items()), 'fixed diagnostic scope changed')
    return cfg


def prepare():
    cfg = configuration()
    out = ROOT / cfg['output']
    require(not out.exists(), 'existing output; verify instead')
    proof = source.verify_report()
    reg, src = source.verify()
    require(src == ROOT / cfg['source'] and proof['report_sha256'] == cfg['source_report_sha256'], 'changed source')
    inputs = dict(reg['inputs'])
    for name in ('registration.json', 'report.json', 'scores.json', 'train.audit.json', 'train.complete.json'):
        p = src / name
        inputs[p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
    for p in (ROOT / CONFIG, Path(__file__), Path(statistics.__file__),
              ROOT / 'tests/evaluation/test_sa_terminal_control_variate.py',
              ROOT / 'docs/SA_TERMINAL_CONTROL_VARIATE_PROTOCOL_ZH.md',
              ROOT / 'docs/SA_ONPOLICY_CROSSFIT_RESULT_ZH.md'):
        inputs[p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
    body = dict(schema=cfg['schema'], config=cfg, inputs=inputs,
        source_binding=reg['binding'], source_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        literature='https://www.jmlr.org/papers/v5/greensmith04a.html',
        adaptation_not_paper_reproduction=True, no_training=True, no_ttf=True, no_promotion=True)
    body['binding'] = run.json_fingerprint(body)
    with source.prior.old.recovery.strict_lock(out, body['binding'], 'terminal-control-variate-prepare'):
        run.once(out / 'registration.json', run.sealed(body))
    return dict(prepared=True, binding=body['binding'], new_solver_calls=0, maximum_parameter_updates=0)


def verify():
    cfg = configuration()
    out = ROOT / cfg['output']
    reg = run.check_seal(run.read_json(out / 'registration.json'))
    require(reg['config'] == cfg and reg['binding'] == run.json_fingerprint(
        {k: v for k, v in reg.items() if k not in ('binding', 'integrity')}), 'registration mismatch')
    for name, sha in reg['inputs'].items():
        require(run.sha256_file(run.contained_file(ROOT, name, field='control-variate input')) == sha,
                'changed input: ' + name)
    proof = source.verify_report()
    require(proof['report_sha256'] == cfg['source_report_sha256'], 'source evidence changed')
    return reg, out


def load_conditions(reg):
    src_reg, src = source.verify()
    original, _ = source.old_statistics(src_reg)
    scores = run.check_seal(run.read_json(src / 'scores.json'))
    require(scores['binding'] == reg['source_binding'], 'score lineage')
    fresh = scores['rows']
    require(len(original) == 16 and len(fresh) == 64 and
            len({r['episode_id'] for r in original + fresh}) == 80 and
            len({r['rng_stream_id'] for r in original + fresh}) == 80, 'replica inventory')
    groups = []
    for pair in src_reg['config']['pairs']:
        a = sorted([r for r in original if r['pair_id'] == pair], key=lambda r: r['replica'])
        b = sorted([r for r in fresh if r['pair_id'] == pair], key=lambda r: r['replica'])
        require([r['replica'] for r in a] == list(range(4)) and
                [r['replica'] for r in b] == list(range(16)), 'condition completeness')
        for key in ('initial_fingerprint', 'policy_sha256', 'pair_id'):
            require(len({r[key] for r in a + b}) == 1, 'changed paired identity')
        groups.append(dict(pair_id=pair, old=a, fresh=b))
    return groups


def condition_worker(job):
    return job['pair_id'], {lane: statistics.condition_report(job[lane]) for lane in ('old', 'fresh')}


def analyze():
    reg, out = verify()
    with source.prior.old.recovery.strict_lock(out, reg['binding'], 'terminal-control-variate-analysis'):
        require(not (out / 'report.json').exists(), 'existing report; verify instead')
        run.write_json(out / 'run_status.json', dict(status='running', binding=reg['binding']))
        try:
            groups = load_conditions(reg)
            # Four real jobs, up to twenty processes; do not invent extra jobs to fill idle CPUs.
            with ProcessPoolExecutor(max_workers=min(reg['config']['workers'], len(groups))) as pool:
                conditions = dict(pool.map(condition_worker, groups))
            result = statistics.summarize(conditions, reg['config']['minimum_second_moment_reduction'],
                                          reg['config']['maximum_condition_increase'])
            result.update(schema=reg['schema'] + '.report', binding=reg['binding'],
                          source_report_sha256=reg['config']['source_report_sha256'])
            verify()
            run.once(out / 'report.json', run.sealed(result))
            run.write_json(out / 'run_status.json', dict(status='complete', binding=reg['binding'], decision=result['decision']))
        except Exception as error:
            run.write_json(out / 'run_status.json', dict(status='failed', binding=reg['binding'],
                                                       error=type(error).__name__, message=str(error)))
            raise
    return dict(complete=True, report_sha256=run.sha256_file(out / 'report.json'),
                decision=result['decision'], relative_change=result['fresh_condition_equal_second_moment']['relative_change'])


def verify_report():
    reg, out = verify()
    result = run.check_seal(run.read_json(out / 'report.json'))
    require(result['binding'] == reg['binding'] and result['fresh_episodes'] == 64 and
            result['no_training'] and result['no_ttf'] and result['no_promotion'], 'report scope')
    rebuilt = {j['pair_id']: {lane: statistics.condition_report(j[lane]) for lane in ('old', 'fresh')}
               for j in load_conditions(reg)}
    expected = statistics.summarize(rebuilt, reg['config']['minimum_second_moment_reduction'],
                                    reg['config']['maximum_condition_increase'])
    for key, value in expected.items():
        require(result[key] == value, 'changed derived result: ' + key)
    return dict(verified=True, report_sha256=run.sha256_file(out / 'report.json'), decision=result['decision'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare', 'analyze', 'verify'))
    args = p.parse_args()
    if args.phase == 'verify':
        result = verify_report()
    else:
        result = globals()[args.phase]()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
