"""Frozen A acceptance-only paired experiment; no training or native modifications."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_expanded_ttf as source
from scripts import run_sa_completion_independent_ttf as old
from experiments import sa_completion_acceptance as rt
from experiments.sa_raw_selection_fast import _bind

run, first, require = source.run, source.first, source.require
CONFIG = 'configs/sa_completion_acceptance.json'
CODE = (CONFIG, 'scripts/run_sa_completion_acceptance.py', 'experiments/sa_completion_acceptance.py',
        'tests/evaluation/test_sa_completion_acceptance.py', 'docs/SA_COMPLETION_ACCEPTANCE_PROTOCOL_ZH.md')


def config():
    c = run.read_json(ROOT/CONFIG)
    fixed = dict(schema='lns2.sa_completion_acceptance.config.v1', source='build/sa-expanded-four-arm-ttf-v1',
        output='build/sa-completion-acceptance-v1', maps=12, task_variants=['bottleneck_d20','bottleneck_d25'],
        solver_seeds=[281], replicas=[0], comparison_arms=list(rt.ARMS), budget_seconds=120.,
        process_fuse_seconds=240., preflight_fuse_seconds=300., audit_fuse_seconds=900.,
        workers=1, audit_workers=20, max_decisions=None, execution_node_budget=None,
        bootstrap=5000, bootstrap_seed=2026092902, timing_mode=rt.TIMING_MODE, model_sha256=rt.MODEL_SHA,
        no_training=True, automatic_promotion=False, reuse_random_streams=True,
        reused_maps_development_only=True, actor_temperature_input_unchanged=True,
        source_report_sha256='4130102b16d29460cd5de76e2373f1140161af8dd9b90539a6aa3f3dbde0b86a')
    require(c == fixed, 'frozen acceptance scope changed')
    return c


def schedule(jobs):
    jobs = list(jobs)
    groups = {}
    for job in jobs:
        rt.identity(job)
        key = (job['pair_id'], job['replica'])
        g = groups.setdefault(key, {})
        require(job['comparison_arm'] not in g, 'duplicate arm')
        g[job['comparison_arm']] = job
    require(len(groups) == 24, 'expected 24 pairs')
    maps = sorted({j['case']['map_id'] for j in jobs})
    require(len(maps) == 12, 'expected 12 maps')
    result = []
    for _, g in sorted(groups.items()):
        require(set(g) == set(rt.ARMS), 'missing arm')
        a, b = [g[x] for x in rt.ARMS]
        for field in ('case','solver_seed','replica','phase','model','plan','expected_initial',
                      'source_folder','source_files','budget_seconds','timing_mode'):
            require(a[field] == b[field], 'unpaired '+field)
        require(a['solver_seed'] == 281 and a['replica'] == 0, 'unexpected seed')
        v = a['case']['task_variant']
        require(v in ('bottleneck_d20','bottleneck_d25'), 'unexpected task')
        parity = (maps.index(a['case']['map_id']) + int(v == 'bottleneck_d25')) % 2
        for arm in (rt.ARMS if not parity else rt.ARMS[::-1]):
            result.append(dict(g[arm], schedule_index=len(result)))
    for arm in rt.ARMS:
        js = [j for j in result if j['comparison_arm'] == arm]
        require(Counter(j['case']['map_id'] for j in js) == {m:2 for m in maps}, 'map coverage')
        for m in maps:
            require({j['case']['task_variant'] for j in js if j['case']['map_id'] == m}
                    == {'bottleneck_d20','bottleneck_d25'}, 'paired densities')
    return result


def prepare():
    c = config()
    out = ROOT/c['output']
    require(not out.exists(), 'already prepared; inspect/resume')
    r, src = source.verify()
    require(run.sha256_file(src/'report.json') == c['source_report_sha256'], 'source report changed')
    report = run.check_seal(run.read_json(src/'report.json'))
    rows = {row['job_id']:row for row in report['episodes']}
    jobs = []
    inputs = dict(r['inputs'])
    # Selection is purely map/task/seed based, never based on source success or TTF.
    for original in r['jobs']:
        if original['comparison_arm'] != 'completion' or original['solver_seed'] != 281:
            continue
        row = rows[original['job_id']]
        folder = src/'episodes'/original['job_id']
        require(run.check_seal(run.read_json(folder/'result.json')) == row, 'source row changed')
        for name, sha in row['files'].items():
            path = (folder/name).relative_to(ROOT).as_posix()
            require(run.sha256_file(ROOT/path) == sha, 'source artifact changed')
            inputs[path] = sha
        for name in ('result.json',):
            inputs[(folder/name).relative_to(ROOT).as_posix()] = run.sha256_file(folder/name)
        for arm in rt.ARMS:
            j = deepcopy(original)
            j.pop('schedule_index', None)
            j.update(job_id=run.json_fingerprint(['completion-acceptance-v1',original['pair_id'],arm])[:24],
                comparison_arm=arm, acceptance_mode=rt.MODES[arm], output=c['output'],
                source_job_id=original['job_id'], source_folder=folder.relative_to(ROOT).as_posix(),
                source_files=row['files'])
            jobs.append(j)
    for name in CODE + tuple((src/n).relative_to(ROOT).as_posix() for n in ('registration.json','report.json')):
        inputs[name] = run.sha256_file(ROOT/name)
    body = dict(schema='lns2.sa_completion_acceptance.registration.v1', config=c, inputs=inputs,
        jobs=schedule(jobs), runtime_labels=rt.LABELS, source_binding=r['binding'],
        source_commit=run.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    body['binding'] = run.json_fingerprint(body)
    with first.recovery.strict_lock(out, body['binding'], 'acceptance-prepare'):
        run.once(out/'registration.json', run.sealed(body))
    return dict(episodes=48, pairs=24, maps=12, workers=1, audit_workers=20, timing_started=False,
                planning_upper_minutes=96, fuse_upper_minutes=192, safe_stop_after_episode=True)


verify = _bind(old.verify, config=config, schedule=schedule)
phase = _bind(old.phase, verify=verify, rt=rt)
_collect = _bind(first.collect, verify=verify, rt=rt)
collect = _bind(old.collect, verify=verify, _collect=_collect)
_report = _bind(old.report, verify=verify, rt=rt)


def report():
    r, out = verify()
    require(not (out/'report.json').exists(), 'report already exists; verify instead')
    return _report()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare','verify','preflight','collect','resume','audit','report','stop','all'))
    args = p.parse_args()
    if args.phase == 'verify':
        r, _ = verify()
        result = dict(inputs=len(r['inputs']), episodes=len(r['jobs']))
    elif args.phase == 'stop':
        out = ROOT/config()['output']
        require(out.exists(), 'not registered')
        run.write_json(out/'STOP_AFTER_EPISODE', dict(requested=True))
        result = dict(stop_after_current_episode=True)
    elif args.phase in ('preflight','audit'):
        result = phase(args.phase)
    elif args.phase == 'resume':
        result = collect(True)
    elif args.phase == 'all':
        result = collect()
        if not result.get('paused'):
            phase('audit')
            result = report()
    else:
        result = globals()[args.phase]()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
