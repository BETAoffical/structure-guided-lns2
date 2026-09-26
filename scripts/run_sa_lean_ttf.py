"""Fresh five-arm collection with scientific checks outside the search clock."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_shared_feature_ttf as old
from scripts import run_sa_onpolicy as run
from experiments import sa_lean_timing as rt
from experiments.sa_raw_selection_fast import _bind

CONFIG = 'configs/sa_lean_ttf.json'
CODE = (CONFIG, 'scripts/run_sa_lean_ttf.py', 'experiments/sa_lean_timing.py',
        'tests/evaluation/test_sa_lean_timing.py', 'docs/SA_LEAN_TIMING_ZH.md')
SOURCE = 'build/sa-shared-feature-ttf-v1'
require = run.require


def config():
    c = run.read_json(ROOT/CONFIG)
    expected = old.config() | dict(schema='lns2.sa_lean_ttf.config.v1',
        output='build/sa-lean-ttf-v1', timing_mode=rt.TIMING_MODE,
        source_report_sha256='ad815c22a65eee5c9e125d811f40c06e3d77ea3a6ed0c5691bb9b05cb1367c21')
    require(c == expected, 'lean timing scope changed')
    return c


def selected_jobs(jobs, c):
    return [dict(j, job_id=run.json_fingerprint(['lean-ttf-v1', j['job_id']])[:24],
                 output=c['output'], timing_mode=rt.TIMING_MODE,
                 source_shared_job_id=j['job_id']) for j in jobs]


def prepare():
    c = config()
    source, source_out = old.verify()
    require(run.sha256_file(source_out/'report.json') == c['source_report_sha256'], 'source report changed')
    for name in ('collect', 'audit', 'pair'):
        old.first.source.check_complete(source, source_out, name, old.phase_jobs(source, name))
    inputs = dict(source['inputs'])
    for name in CODE + (SOURCE+'/registration.json', SOURCE+'/report.json'):
        inputs[name] = run.sha256_file(run.contained_file(ROOT, name, field='lean input'))
    body = dict(schema='lns2.sa_lean_ttf.registration.v1', config=c, inputs=inputs,
                jobs=selected_jobs(source['jobs'], c), runtime_labels=rt.LABELS,
                source_binding=source['binding'], no_training=True, no_promotion=True,
                source_commit=run.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    body['binding'] = run.json_fingerprint(body)
    out = ROOT/c['output']
    require(not out.exists(), 'already registered; verify/resume instead')
    with old.first.recovery.strict_lock(out, body['binding'], 'lean-prepare'):
        run.once(out/'registration.json', run.sealed(body))
    return dict(episodes=len(body['jobs']), conditions=48, maps=6, workers=1,
                timing_started=False, timing_mode=rt.TIMING_MODE)


def verify():
    c = config()
    out = ROOT/c['output']
    r = run.check_seal(run.read_json(out/'registration.json'))
    require(r['schema'] == 'lns2.sa_lean_ttf.registration.v1' and r['config'] == c and
            r['runtime_labels'] == rt.LABELS and r['binding'] == run.json_fingerprint(
                {k: v for k, v in r.items() if k not in ('binding', 'integrity')}), 'lean registration changed')
    for name, digest in r['inputs'].items():
        require(run.sha256_file(run.contained_file(ROOT, name, field='lean input')) == digest,
                'changed lean timing input: '+name)
    source = run.check_seal(run.read_json(ROOT/SOURCE/'registration.json'))
    require(r['source_binding'] == source['binding'] and r['jobs'] == selected_jobs(source['jobs'], c),
            'lean schedule/model/stream changed')
    return r, out


phase_jobs = old.phase_jobs
phase = _bind(old.phase, verify=verify, rt=rt)
collect = _bind(old.collect, verify=verify, rt=rt)


def report():
    r, out = verify()
    jobs = old.first.source.jobs_for(r)
    for name in ('collect', 'audit', 'pair'):
        old.first.source.check_complete(r, out, name, phase_jobs(r, name))
    rows = [old.first.source.read_result(r, out, j) for j in jobs]
    for j in jobs:
        proof = run.check_seal(run.read_json(out/'audit'/(j['job_id']+'.json')))
        require(proof['result_sha256'] == run.sha256_file(out/'episodes'/j['job_id']/'result.json'), 'stale audit')
    for j in phase_jobs(r, 'pair'):
        proof = run.check_seal(run.read_json(out/'pair'/(j['job_id']+'.json')))
        require(proof['result_hashes'] == [run.sha256_file(out/'episodes'/x/'result.json') for x in j['runtime_jobs']],
                'stale runtime pair')
    result = rt.summarize(rows, r['config']['bootstrap'], r['config']['bootstrap_seed'])
    cases = {j['case']['task_id']: j['case']['task_variant'] for j in jobs}
    result.update(schema='lns2.sa_lean_ttf.report.v1', binding=r['binding'], episodes=rows,
                  by_density={v: rt.summarize([x for x in rows if cases[x['task_id']] == v],
                      r['config']['bootstrap'], r['config']['bootstrap_seed'])
                      for v in ('bottleneck_d20', 'bottleneck_d25')})
    run.once(out/'report.json', run.sealed(result))
    run.write_json(out/'run_status.json', dict(status='complete', completed=len(jobs), total=len(jobs), binding=r['binding']))
    return {k: v for k, v in result.items() if k not in ('episodes', 'by_map', 'by_density')}


main = _bind(old.main, config=config, prepare=prepare, verify=verify, phase=phase,
             collect=collect, report=report, __doc__=__doc__)

if __name__ == '__main__':
    main()
