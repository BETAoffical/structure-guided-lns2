"""Paired, serial parent/A timing; safe stop after each episode."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_terminal_efficiency as training
from scripts import run_sa_lean_ttf as lean
from scripts import run_sa_raw_fast_ttf as first
from scripts import run_sa_shared_feature_ttf as shared
from scripts import run_sa_onpolicy as run
from experiments import sa_terminal_efficiency_timing as rt
from experiments.sa_raw_selection_fast import _bind

CONFIG = 'configs/sa_terminal_efficiency_ttf.json'
CODE = (CONFIG, 'scripts/run_sa_terminal_efficiency_ttf.py',
        'experiments/sa_terminal_efficiency_timing.py',
        'tests/evaluation/test_sa_terminal_efficiency_timing.py',
        'docs/SA_TERMINAL_EFFICIENCY_TTF_PROTOCOL_ZH.md')
require = run.require


def config():
    c = run.read_json(ROOT/CONFIG)
    fixed = dict(comparison_arms=list(rt.ARMS), solver_seeds=[251, 257], replicas=[0, 1],
                 budget_seconds=120., process_fuse_seconds=240., workers=1, audit_workers=20,
                 preflight_steps=3, preflight_fuse_seconds=300., audit_fuse_seconds=900.,
                 max_decisions=None, execution_node_budget=None, bootstrap=5000,
                 bootstrap_seed=2026092702, timing_mode=rt.TIMING_MODE,
                 no_training=True, automatic_promotion=False)
    require(all(c[k] == v for k, v in fixed.items()), 'terminal timing scope changed')
    require(c['schema'] == 'lns2.sa_terminal_efficiency_ttf.config.v1' and
            c['output'] == 'build/sa-terminal-efficiency-ttf-v1' and
            c['source'] == 'build/sa-terminal-efficiency-v1', 'terminal timing identity changed')
    return c


def schedule(jobs):
    groups = {}
    for job in jobs:
        g = groups.setdefault((job['pair_id'], job['replica']), {})
        require(job['comparison_arm'] not in g, 'duplicate timing arm')
        g[job['comparison_arm']] = job
    require(len(groups) == 48, 'expected all 48 paired positions')
    result = []
    for i, (_, g) in enumerate(sorted(groups.items())):
        require(set(g) == set(rt.ARMS), 'missing paired model')
        for field in ('expected_initial', 'phase', 'solver_seed'):
            require(g['parent'][field] == g['completion'][field], 'unpaired '+field)
        require(g['parent']['case'] == g['completion']['case'], 'unpaired task')
        require(g['parent']['plan']['config']['stream_seed'] ==
                g['completion']['plan']['config']['stream_seed'], 'unpaired stream')
        for arm in (rt.ARMS if i % 2 == 0 else rt.ARMS[::-1]):
            result.append(dict(g[arm], schedule_index=len(result)))
    for arm in rt.ARMS:
        subset = [j for j in result if j['comparison_arm'] == arm]
        require(sorted(Counter(j['case']['map_id'] for j in subset).values()) == [8]*6, 'six map coverage')
        for m in {j['case']['map_id'] for j in subset}:
            actual = {(j['case']['task_variant'], j['solver_seed'], j['replica'])
                      for j in subset if j['case']['map_id'] == m}
            expected = {(v, s, r) for v in ('bottleneck_d20', 'bottleneck_d25')
                        for s in (251, 257) for r in (0, 1)}
            require(actual == expected, 'incomplete map/task/seed coverage')
    return result


def selected_jobs(reg, out, c):
    jobs = []
    for source, row in training.audited_rows(reg, out, 'comparison'):
        if source['comparison_arm'] not in rt.ARMS:
            continue
        j = deepcopy(source)
        require(j['model']['sha256'] == c[j['comparison_arm']+'_sha256'], 'wrong frozen model')
        j.pop('parent_pid', None)
        j.update(source_job_id=source['job_id'],
                 source_folder=training.previous.folder(source).relative_to(ROOT).as_posix(),
                 source_files=row['files'], source_decisions=row['decisions'],
                 job_id=run.json_fingerprint(['terminal-efficiency-ttf-v1', source['job_id']])[:24],
                 output=c['output'], budget_seconds=c['budget_seconds'], preflight_steps=c['preflight_steps'],
                 timing_mode=rt.TIMING_MODE, max_decisions=None, execution_node_budget=None)
        jobs.append(j)
    return schedule(jobs)


def prepare():
    c = config()
    src, src_out = training.verify()
    previous, _ = lean.verify()
    require(run.sha256_file(src_out/'report.json') == c['source_report_sha256'], 'source report changed')
    report = run.check_seal(run.read_json(src_out/'report.json'))
    require(report['development_signal']['completion'] and report['no_ttf'], 'no registered A signal')
    jobs = selected_jobs(src, src_out, c)
    inputs = dict(previous['inputs'])
    for path, sha in src['inputs'].items():
        require(path not in inputs or inputs[path] == sha, 'inconsistent frozen input: '+path)
        inputs[path] = sha
    paths = list(CODE) + ['configs/sa_lean_ttf.json', 'scripts/run_sa_lean_ttf.py',
                         'build/sa-lean-ttf-v1/registration.json', 'build/sa-lean-ttf-v1/report.json']
    paths += [(src_out/n).relative_to(ROOT).as_posix() for n in
              ('registration.json', 'report.json', 'comparison.complete.json', 'comparison.audit.json',
               'update.completion.json', 'models/completion.json', 'parity.windows.json', 'parity.wsl.json')]
    for j in jobs:
        for name, sha in j['source_files'].items():
            path = j['source_folder']+'/'+name
            require(run.sha256_file(ROOT/path) == sha, 'changed source prefix')
            inputs[path] = sha
    for path in paths:
        inputs[path] = run.sha256_file(run.contained_file(ROOT, path, field='terminal timing input'))
    body = dict(schema='lns2.sa_terminal_efficiency_ttf.registration.v1', config=c, inputs=inputs, jobs=jobs,
                runtime_labels=rt.LABELS, source_binding=src['binding'], no_training=True, no_promotion=True,
                source_commit=run.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    body['binding'] = run.json_fingerprint(body)
    out = ROOT/c['output']
    require(not out.exists(), 'already registered; verify/resume')
    with first.recovery.strict_lock(out, body['binding'], 'terminal-timing-prepare'):
        run.once(out/'registration.json', run.sealed(body))
    return dict(episodes=96, paired_positions=48, maps=6, workers=1, audit_workers=20,
                planning_budget_upper_hours=3.2, process_fuse_upper_hours=6.4,
                safe_stop_after_episode=True, timing_started=False)


def verify():
    c = config()
    out = ROOT/c['output']
    r = run.check_seal(run.read_json(out/'registration.json'))
    require(r['config'] == c and r['runtime_labels'] == rt.LABELS and r['binding'] == run.json_fingerprint(
        {k: v for k, v in r.items() if k not in ('binding', 'integrity')}), 'timing registration changed')
    for path, sha in r['inputs'].items():
        require(run.sha256_file(run.contained_file(ROOT, path, field='terminal timing input')) == sha,
                'changed terminal timing input: '+path)
    src, src_out = training.verify()
    require(r['source_binding'] == src['binding'] and r['jobs'] == selected_jobs(src, src_out, c),
            'timing model/schedule/stream changed')
    return r, out


def phase_jobs(r, name):
    return [] if name == 'pair' else first.source.jobs_for(r)


def phase(name):
    if name == 'pair':
        return dict(skipped=True, reason='different models; engineering equivalence already verified')
    require(name in ('preflight', 'audit'), 'unknown timing phase')
    return _bind(shared.phase, verify=verify, rt=rt, phase_jobs=phase_jobs)(name)


collect = _bind(first.collect, verify=verify, rt=rt)


def report():
    r, out = verify()
    jobs = first.source.jobs_for(r)
    for name in ('collect', 'audit'):
        first.source.check_complete(r, out, name, jobs)
    rows = [first.source.read_result(r, out, j) for j in jobs]
    for j in jobs:
        proof = run.check_seal(run.read_json(out/'audit'/(j['job_id']+'.json')))
        require(proof['result_sha256'] == run.sha256_file(out/'episodes'/j['job_id']/'result.json'), 'stale audit')
    cases = {j['case']['task_id']: j['case']['task_variant'] for j in jobs}
    result = rt.summarize(rows, r['config']['bootstrap'], r['config']['bootstrap_seed'])
    result.update(schema='lns2.sa_terminal_efficiency_ttf.report.v1', binding=r['binding'], episodes=rows,
                  by_density={v: rt.summarize([x for x in rows if cases[x['task_id']] == v],
                      r['config']['bootstrap'], r['config']['bootstrap_seed'])
                      for v in ('bottleneck_d20', 'bottleneck_d25')})
    run.once(out/'report.json', run.sealed(result))
    run.write_json(out/'run_status.json', dict(status='complete', completed=len(jobs), total=len(jobs), binding=r['binding']))
    return {k: v for k, v in result.items() if k not in ('episodes', 'by_map', 'by_density')}


main = _bind(first.main, config=config, prepare=prepare, verify=verify, phase=phase,
             collect=collect, report=report, __doc__=__doc__)

if __name__ == '__main__':
    main()
