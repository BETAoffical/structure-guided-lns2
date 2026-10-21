"""Paired frozen-A prior ablation; no training, native edits or default changes."""
import argparse
from collections import Counter
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_expanded_ttf as source
from scripts import run_sa_completion_independent_ttf as old
from experiments import sa_completion_prior as rt
from experiments.sa_raw_selection_fast import _bind

run, first, require = source.run, source.first, source.require
CONFIG = 'configs/sa_completion_prior.json'
CODE = (CONFIG, 'scripts/run_sa_completion_prior.py', 'experiments/sa_completion_prior.py',
        'tests/evaluation/test_sa_completion_prior.py', 'docs/SA_COMPLETION_PRIOR_PROTOCOL_ZH.md',
        'experiments/sa_completion_acceptance.py')


def config():
    c = run.read_json(ROOT/CONFIG)
    fixed = dict(schema='lns2.sa_completion_prior.config.v1', source='build/sa-expanded-four-arm-ttf-v1',
        output='build/sa-completion-prior-v1', maps=12, task_variants=['bottleneck_d20','bottleneck_d25'],
        solver_seeds=[281], replicas=[0], comparison_arms=list(rt.ARMS), budget_seconds=120.,
        process_fuse_seconds=240., preflight_fuse_seconds=300., audit_fuse_seconds=900., workers=1,
        audit_workers=4, max_decisions=None, execution_node_budget=None, bootstrap=5000,
        bootstrap_seed=2026100401, timing_mode=rt.TIMING_MODE, model_sha256=rt.MODEL_SHA,
        no_training=True, automatic_promotion=False, reuse_random_streams=True,
        reused_maps_development_only=True, score_feature_unchanged=True,
        source_report_sha256='4130102b16d29460cd5de76e2373f1140161af8dd9b90539a6aa3f3dbde0b86a')
    require(c == fixed, 'frozen prior scope changed')
    return c


def schedule(jobs):
    groups = {}
    for j in jobs:
        rt.identity(j)
        g = groups.setdefault((j['pair_id'],j['replica']), {})
        require(j['comparison_arm'] not in g, 'duplicate arm')
        g[j['comparison_arm']] = j
    require(len(groups) == 24, 'expected 24 conditions')
    maps = sorted({g['original']['case']['map_id'] for g in groups.values() if 'original' in g})
    require(len(maps) == 12, 'expected 12 maps')
    orders = list(itertools.permutations(rt.ARMS))
    result = []
    for _, g in sorted(groups.items()):
        require(set(g) == set(rt.ARMS), 'missing arm')
        a = g['original']
        for field in ('case','solver_seed','replica','phase','model','plan','expected_initial',
                      'source_folder','source_files','budget_seconds','timing_mode'):
            require(all(g[x][field] == a[field] for x in rt.ARMS), 'unpaired '+field)
        require(a['solver_seed'] == 281 and a['replica'] == 0, 'unexpected seed')
        variant = a['case']['task_variant']
        require(variant in ('bottleneck_d20','bottleneck_d25'), 'unexpected density')
        order = orders[(maps.index(a['case']['map_id']) + (3 if variant == 'bottleneck_d25' else 0)) % 6]
        for arm in order:
            result.append(dict(g[arm], schedule_index=len(result)))
    for arm in rt.ARMS:
        js = [j for j in result if j['comparison_arm'] == arm]
        require(Counter(j['case']['map_id'] for j in js) == {m:2 for m in maps}, 'map coverage')
        for m in maps:
            require({j['case']['task_variant'] for j in js if j['case']['map_id'] == m}
                    == {'bottleneck_d20','bottleneck_d25'}, 'density coverage')
    return result


def prepare():
    c = config()
    out = ROOT/c['output']
    require(not out.exists(), 'already registered; inspect/resume')
    r, src = source.verify()
    require(run.sha256_file(src/'report.json') == c['source_report_sha256'], 'source report changed')
    rows = {x['job_id']:x for x in run.check_seal(run.read_json(src/'report.json'))['episodes']}
    inputs, jobs = dict(r['inputs']), []
    for original in r['jobs']:
        if original['comparison_arm'] != 'completion' or original['solver_seed'] != 281 or original['replica'] != 0:
            continue
        row = rows[original['job_id']]
        folder = src/'episodes'/original['job_id']
        require(run.check_seal(run.read_json(folder/'result.json')) == row, 'source result changed')
        for name, sha in row['files'].items():
            path = (folder/name).relative_to(ROOT).as_posix()
            require(run.sha256_file(ROOT/path) == sha, 'source artifact changed')
            inputs[path] = sha
        inputs[(folder/'result.json').relative_to(ROOT).as_posix()] = run.sha256_file(folder/'result.json')
        for arm in rt.ARMS:
            j = deepcopy(original)
            j.pop('schedule_index', None)
            j.update(job_id=run.json_fingerprint(['completion-prior-v1',original['pair_id'],arm])[:24],
                     comparison_arm=arm, prior_mode=arm, output=c['output'], source_job_id=original['job_id'],
                     source_folder=folder.relative_to(ROOT).as_posix(), source_files=row['files'])
            jobs.append(j)
    for name in CODE + tuple((src/n).relative_to(ROOT).as_posix() for n in ('registration.json','report.json')):
        inputs[name] = run.sha256_file(ROOT/name)
    # Include the inherited modules even if a historical registration omitted one.
    for name in ('scripts/run_sa_completion_independent_ttf.py','scripts/run_sa_raw_fast_ttf.py',
                 'experiments/sa_lean_timing.py','experiments/sa_shared_features.py',
                 'experiments/sa_raw_selection_fast.py','experiments/sa_raw_timed_runtime.py',
                 'experiments/sa_raw_residual_actor.py','experiments/sa_onpolicy_actor.py',
                 'experiments/sa_onpolicy_contract.py','experiments/sa_terminal_efficiency_timing.py',
                 'experiments/sa_raw_timing_metrics.py'):
        inputs[name] = run.sha256_file(ROOT/name)
    body = dict(schema='lns2.sa_completion_prior.registration.v1', config=c, inputs=inputs,
                jobs=schedule(jobs), runtime_labels=rt.LABELS, source_binding=r['binding'],
                source_commit=run.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    body['binding'] = run.json_fingerprint(body)
    with first.recovery.strict_lock(out,body['binding'],'prior-prepare'):
        run.once(out/'registration.json',run.sealed(body))
    return dict(episodes=72, pairs=24, maps=12, workers=1, audit_workers=4, timing_started=False,
                planning_upper_minutes=144, process_fuse_upper_minutes=288)


verify = _bind(old.verify, config=config, schedule=schedule)
_collect = _bind(first.collect, verify=verify, rt=rt)
collect = _bind(old.collect, verify=verify, _collect=_collect)
_report = _bind(old.report, verify=verify, rt=rt)


def phase(name):
    require(name in ('preflight','audit'), 'unknown phase')
    r, out = verify()
    jobs = first.source.jobs_for(r)
    if name == 'audit':
        first.source.check_complete(r,out,'collect',jobs)
    with first.recovery.strict_lock(out,r['binding'],'prior-'+name):
        require(not (out/name).exists(), 'phase already attempted; inspect')
        rows = first.source.execute(r,out,jobs,rt.preflight_worker if name == 'preflight' else rt.audit_worker,
            name,r['config']['audit_workers'],r['config']['preflight_fuse_seconds' if name == 'preflight' else 'audit_fuse_seconds'])
        run.once(out/(name+'.complete.json'),run.sealed(dict(binding=r['binding'],jobs=len(jobs),
            files={j['job_id']:run.sha256_file(out/name/(j['job_id']+'.json')) for j in jobs})))
    return dict(phase=name,verified=len(rows))


def report():
    r, out = verify()
    require(not (out/'report.json').exists(), 'report already exists; verify instead')
    result = _report()
    behavior = {}
    for arm in rt.ARMS:
        proofs = [run.check_seal(run.read_json(out/'audit'/(j['job_id']+'.json')))
                  for j in r['jobs'] if j['comparison_arm'] == arm]
        count = sum(p['decisions'] for p in proofs)
        hits = sum(p['true_anchor_hits'] for p in proofs)
        mass = sum(p['true_anchor_probability_sum'] for p in proofs)
        behavior[arm] = dict(decisions=count,true_anchor_hits=hits,
                            true_anchor_selection_fraction=hits/count if count else None,
                            decision_weighted_mean_true_anchor_probability=mass/count if count else None)
    run.once(out/'behavior.json',run.sealed(dict(binding=r['binding'],arms=behavior,
        descriptive_different_visited_states=True,source_score_retained=True)))
    return result | dict(behavior=behavior)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','verify','preflight','collect','resume','audit','report','stop','all'))
    args = p.parse_args()
    if args.phase == 'verify':
        r, _ = verify()
        value = dict(inputs=len(r['inputs']),episodes=len(r['jobs']))
    elif args.phase == 'stop':
        out = ROOT/config()['output']
        require(out.exists(), 'not registered')
        run.write_json(out/'STOP_AFTER_EPISODE',dict(requested=True))
        value = dict(stop_after_current_episode=True)
    elif args.phase in ('preflight','audit'):
        value = phase(args.phase)
    elif args.phase == 'resume':
        value = collect(True)
    elif args.phase == 'all':
        value = collect()
        if not value.get('paused'):
            phase('audit')
            value = report()
    else:
        value = globals()[args.phase]()
    print(json.dumps(value,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
