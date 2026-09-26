"""Authorized serial TTF of existing fast/shared runtimes; safe episode-boundary stop."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_raw_fast_ttf as first
from scripts import audit_sa_shared_features as engineering
from scripts import run_sa_onpolicy as run
from experiments import sa_shared_feature_timing as rt
from experiments.sa_raw_selection_fast import _bind

CONFIG = 'configs/sa_shared_feature_ttf.json'
CODE = (CONFIG, 'scripts/run_sa_shared_feature_ttf.py', 'experiments/sa_shared_feature_timing.py',
        'tests/evaluation/test_sa_shared_feature_timing.py', 'docs/SA_SHARED_FEATURE_TTF_PROTOCOL_ZH.md')
require = run.require


def config():
    c = run.read_json(ROOT/CONFIG)
    expected = dict(solver_seeds=[251, 257], replicas=[0, 1], budget_seconds=120., process_fuse_seconds=240.,
                    workers=1, audit_workers=20, preflight_steps=3, preflight_fuse_seconds=300.,
                    audit_fuse_seconds=900., max_decisions=None, execution_node_budget=None,
                    bootstrap=5000, bootstrap_seed=2026092617, no_training=True, automatic_promotion=False)
    require(all(c[k] == v for k, v in expected.items()), 'shared timing scope changed')
    require(c['schema'] == 'lns2.sa_shared_feature_ttf.config.v1' and
            c['output'] == 'build/sa-shared-feature-ttf-v1', 'shared timing identity changed')
    return c


def selected_jobs(old_jobs, c):
    groups = {}
    for old in old_jobs:
        if old['solver_seed'] not in c['solver_seeds'] or old['replica'] not in c['replicas']:
            continue
        names = {'raw_updated': ('raw_reference', 'raw_fast'), 'dual16_sa': ('dual16_sa',),
                 'official_sa': ('official_sa', 'official')}.get(old['comparison_arm'], ())
        for name in names:
            j = deepcopy(old)
            j.update(source_timed_job_id=old['job_id'], comparison_arm=name, output=c['output'],
                     job_id=run.json_fingerprint(['shared-ttf-v1', old['job_id'], name])[:24],
                     runtime_variant={'raw_reference': 'existing_fast', 'raw_fast': 'shared_features',
                                      'dual16_sa': 'reference', 'official_sa': 'annealed',
                                      'official': 'standard'}[name])
            if name == 'official':
                j['arm'] = 'official'
            g = groups.setdefault((j['pair_id'], j['replica']), {})
            require(name not in g, 'duplicate source condition')
            g[name] = j
    require(len(groups) == 48, 'expected 48 paired conditions')
    result = []
    # Ten Williams orders balance preceding treatments and positions over each cycle.
    base = [0, 1, 4, 2, 3]
    orders = [tuple(rt.ARMS[(x+shift) % 5] for x in order)
              for order in (base, base[::-1]) for shift in range(5)]
    for i, (_, g) in enumerate(sorted(groups.items())):
        require(set(g) == set(rt.ARMS), 'missing source arm')
        for name in orders[i % len(orders)]:
            result.append(dict(g[name], schedule_index=len(result)))
    for name in rt.ARMS:
        arm = [j for j in result if j['comparison_arm'] == name]
        require(sorted(Counter(j['case']['map_id'] for j in arm).values()) == [8]*6, 'six map coverage')
        for m in {j['case']['map_id'] for j in arm}:
            actual = {(j['case']['task_variant'], j['solver_seed'], j['replica']) for j in arm if j['case']['map_id'] == m}
            expected = {(v, s, r) for v in ('bottleneck_d20', 'bottleneck_d25')
                        for s in c['solver_seeds'] for r in c['replicas']}
            require(actual == expected, 'map/task/seed coverage')
    return result


def prepare():
    c = config()
    er, eo = engineering.registered(engineering.OUTPUT)
    require(run.sha256_file(eo/'report.json') == c['engineering_report_sha256'], 'engineering report changed')
    for phase in ('profile', 'verify', 'benchmark', 'native'):
        jobs = er['jobs'] if phase == 'verify' else engineering.old.sample_jobs(er['jobs'])
        first.source.check_complete(er, eo, phase, jobs)
    src = ROOT/engineering.SOURCE
    old = run.check_seal(run.read_json(src/'registration.json'))
    require(run.sha256_file(src/'report.json') == c['source_report_sha256'], 'source report changed')
    jobs = selected_jobs(old['jobs'], c)
    out = ROOT/c['output']
    require(not out.exists(), 'already registered; verify/resume')
    inputs = dict(er['inputs'])
    def bind(name, expected=None):
        digest = run.sha256_file(run.contained_file(ROOT, name, field='shared timing input'))
        require(expected is None or digest == expected, 'changed frozen artifact: '+name)
        inputs[name] = digest
    for name in CODE:
        bind(name)
    for name in ('registration.json', 'report.json', 'profile.complete.json', 'verify.complete.json',
                 'benchmark.complete.json', 'native.complete.json'):
        bind((eo/name).relative_to(ROOT).as_posix())
    for j in jobs:
        require(j['budget_seconds'] == c['budget_seconds'] and j['preflight_steps'] == c['preflight_steps'],
                'source budget or prefix changed')
        for name, digest in j['source_files'].items():
            bind(j['source_folder']+'/'+name, digest)
        bind(j['plan']['native_file'], j['plan']['config']['native_sha256'])
        if j['model']:
            bind(j['model']['path'], j['model']['sha256'])
    body = dict(schema='lns2.sa_shared_feature_ttf.registration.v1', config=c, inputs=inputs, jobs=jobs,
                engineering_binding=er['binding'], runtime_labels=rt.LABELS, no_training=True, no_promotion=True,
                source_commit=run.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    body['binding'] = run.json_fingerprint(body)
    with first.recovery.strict_lock(out, body['binding'], 'shared-timing-prepare'):
        run.once(out/'registration.json', run.sealed(body))
    return dict(episodes=240, conditions=48, maps=6, workers=1, audit_workers=20,
                planning_budget_upper_minutes=480, process_fuse_upper_minutes=960,
                runtime_labels=rt.LABELS, safe_stop_after_episode=True)


def verify():
    c = config()
    out = ROOT/c['output']
    r = run.check_seal(run.read_json(out/'registration.json'))
    require(r['config'] == c and r['runtime_labels'] == rt.LABELS and r['binding'] == run.json_fingerprint(
        {k: v for k, v in r.items() if k not in ('binding', 'integrity')}), 'registration changed')
    for name, digest in r['inputs'].items():
        require(run.sha256_file(run.contained_file(ROOT, name, field='shared timing input')) == digest,
                'changed shared timing input: '+name)
    old = run.check_seal(run.read_json(ROOT/engineering.SOURCE/'registration.json'))
    require(r['jobs'] == selected_jobs(old['jobs'], c), 'schedule/model/stream identity changed')
    return r, out


def phase_jobs(r, name):
    jobs = first.source.jobs_for(r)
    if name != 'pair':
        return jobs
    groups = {}
    for j in jobs:
        groups.setdefault((j['pair_id'], j['replica']), {})[j['comparison_arm']] = j
    return [dict(g['raw_fast'], runtime_jobs=[g[a]['job_id'] for a in ('raw_reference', 'raw_fast')])
            for _, g in sorted(groups.items())]


def phase(name):
    r, out = verify()
    jobs = phase_jobs(r, name)
    if name != 'preflight':
        first.source.check_complete(r, out, 'collect', first.source.jobs_for(r))
    worker = dict(preflight=rt.preflight_worker, audit=rt.audit_worker, pair=rt.pair_worker)[name]
    with first.recovery.strict_lock(out, r['binding'], 'shared-'+name):
        require(not (out/name).exists(), 'phase already attempted; inspect')
        rows = first.source.execute(r, out, jobs, worker, name, r['config']['audit_workers'],
                                    r['config']['preflight_fuse_seconds' if name == 'preflight' else 'audit_fuse_seconds'])
        require(len(rows) == len(jobs), 'incomplete phase')
        run.once(out/(name+'.complete.json'), run.sealed(dict(binding=r['binding'], jobs=len(jobs),
                 files={j['job_id']: run.sha256_file(out/name/(j['job_id']+'.json')) for j in jobs})))
    return dict(phase=name, verified=len(rows))


# The tested collector and all timing/stop/resume boundaries are retained.
collect = _bind(first.collect, verify=verify, rt=rt)


def report():
    r, out = verify()
    jobs = first.source.jobs_for(r)
    for name in ('collect', 'audit', 'pair'):
        first.source.check_complete(r, out, name, phase_jobs(r, name))
    rows = [first.source.read_result(r, out, j) for j in jobs]
    for j in jobs:
        proof = run.check_seal(run.read_json(out/'audit'/(j['job_id']+'.json')))
        require(proof['result_sha256'] == run.sha256_file(out/'episodes'/j['job_id']/'result.json'), 'stale audit')
    for j in phase_jobs(r, 'pair'):
        proof = run.check_seal(run.read_json(out/'pair'/(j['job_id']+'.json')))
        require(proof['result_hashes'] == [run.sha256_file(out/'episodes'/x/'result.json') for x in j['runtime_jobs']],
                'stale pair audit')
    result = rt.summarize(rows, r['config']['bootstrap'], r['config']['bootstrap_seed'])
    cases = {j['case']['task_id']: j['case']['task_variant'] for j in jobs}
    result.update(schema='lns2.sa_shared_feature_ttf.report.v1', binding=r['binding'], episodes=rows,
                  by_density={v: rt.summarize([x for x in rows if cases[x['task_id']] == v],
                                  r['config']['bootstrap'], r['config']['bootstrap_seed'])
                              for v in ('bottleneck_d20', 'bottleneck_d25')})
    run.once(out/'report.json', run.sealed(result))
    run.write_json(out/'run_status.json', dict(status='complete', completed=len(jobs), total=len(jobs), binding=r['binding']))
    return {k: v for k, v in result.items() if k not in ('episodes', 'by_map', 'by_density')}


main = _bind(first.main, config=config, prepare=prepare, verify=verify, phase=phase, collect=collect, report=report)

if __name__ == '__main__':
    main()
