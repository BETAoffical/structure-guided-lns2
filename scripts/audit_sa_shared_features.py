"""Replay frozen inputs and benchmark shared features, never collect TTF or labels."""
import argparse
import cProfile
from copy import deepcopy
import json
import os
from pathlib import Path
import statistics
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_sa_raw_selection_cost as old
from scripts import run_sa_onpolicy as run
from experiments.online_feature_engine import OnlineFeatureEngine, TopologyAnalysisCache
from experiments.sa_raw_selection_fast import FastPolicy
from experiments.sa_shared_features import SharedFeatureEngine, SharedFeaturePolicy
from lns2_selector.runtime.online_selection import score_online_candidates

SOURCE = 'build/sa-raw-serial-ttf-v1'
OUTPUT = 'build/sa-shared-features-v1'
require = run.require


class RecordedFeaturePool:
    """Recompute GBDT features/scores over archived proposals, without proposing."""

    def __init__(self, model, shared):
        self.model, self.shared = model, shared
        self.engine = self.topology = self.event = None

    def select(self, env, state, decision):
        index, scored = old.SavedPool(self.event).select(env, state, decision)
        candidates = [{k: v for k, v in c.items() if k != 'score'} for c in scored]
        if self.engine is None:
            kind = SharedFeatureEngine if self.shared else OnlineFeatureEngine
            self.engine = kind(state, backend='native', dense_output=True,
                               required_features={'realized_dynamic': set(self.model.base_feature_names)})
        if self.topology is None:
            self.topology = TopologyAnalysisCache(state, static_grid=self.engine.static_grid, backend='native')
        else:
            self.topology.prepare(state)
        self.engine.prepare(state, prepared_native_analysis=self.topology.last_native_prepared)
        rows, _ = self.engine.realized_rows(candidates, state_hash=self.event['before'])
        actual_index, scores, _ = score_online_candidates(rows, self.model)
        require(actual_index == index and scores == [c['score'] for c in scored], 'GBDT score/anchor drift')
        return index, scored


def policy_for(kind, job, q, initial):
    policy = (SharedFeaturePolicy if kind == 'shared' else FastPolicy)(job, q, old.context(job))
    policy.selector = RecordedFeaturePool(policy.selector.model, kind == 'shared')
    policy.start(initial)
    return policy


def worker(job):
    q, folder, row, initial = old.loaded(job)
    policy = policy_for('shared', job, q, initial)
    state, previous = initial, initial
    count = candidates = 0
    samples = []
    wanted = set(old.indices(row['decisions']))
    for event in run.trace_read(folder):
        d = event['decision']
        require(d == count and event['before'] == q.state_fingerprint(state), 'trace continuity')
        if job['mode'] == 'verify':
            policy.selector.event = event
            old.equal_event(policy.choose(None, state, d), event)
            candidates += len(event['pool'])
        elif d in wanted:
            observations = []
            rounds = 1 if job['mode'] == 'profile' else 3
            for repeat in range(rounds):
                order = ['reference', 'shared']
                if (job['sample_index'] + d + repeat) % 2:
                    order.reverse()
                for kind in order:
                    p = policy_for(kind, job, q, initial)
                    p.history = deepcopy(policy.history)
                    p.engine.prepare(previous)
                    p.selector.event = event
                    # Populate the selector's prior-state cache outside the timed region.
                    engine = SharedFeatureEngine if kind == 'shared' else OnlineFeatureEngine
                    p.selector.engine = engine(previous, backend='native', dense_output=True,
                        required_features={'realized_dynamic': set(p.selector.model.base_feature_names)})
                    p.selector.topology = TopologyAnalysisCache(previous,
                        static_grid=p.selector.engine.static_grid, backend='native')
                    snapshot = deepcopy(state)
                    if job['mode'] == 'profile':
                        profiler = cProfile.Profile()
                        answer = profiler.runcall(p.choose, None, snapshot, d)
                        measured = dict(profile=old.profile_rows(profiler))
                    else:
                        began = time.perf_counter()
                        answer = p.choose(None, snapshot, d)
                        measured = dict(seconds=time.perf_counter() - began)
                    old.equal_event(answer, event)
                    require(q.state_fingerprint(snapshot) == event['before'], 'state mutated')
                    observations.append(dict(mode=kind, repeat=repeat, **measured))
            samples.append(dict(decision=d, candidates=len(event['pool']), observations=observations))
        after = q.apply_state_delta(state, event['delta'])
        policy.observe(state, event, after)
        previous, state = state, after
        count += 1
    require(count == row['decisions'] and q.state_fingerprint(state) == row['final_fingerprint'], 'final state drift')
    return dict(status='ok', job_id=job['job_id'], map_id=job['case']['map_id'],
                decisions=count, candidates=candidates, samples=samples, no_new_repairs=True)


def native_worker(job):
    # Reuse the archived-path/PP audit, replacing only its explicitly opted-in policy.
    from experiments.sa_raw_selection_fast import _bind
    return _bind(old.native_worker, FastPolicy=SharedFeaturePolicy)(job)


def prepare(output):
    out, src = ROOT/output, ROOT/SOURCE
    require(not out.exists(), 'existing audit; inspect or resume, never overwrite')
    registration = run.check_seal(run.read_json(src/'registration.json'))
    require(run.sha256_file(src/'report.json') == old.SOURCE_SHA, 'changed formal report')
    for phase in ('collect', 'audit'):
        old.source.check_complete(registration, src, phase, registration['jobs'])
    jobs = []
    inputs = {}
    def bind(name, expected=None):
        path = run.contained_file(ROOT, name, field='shared feature input')
        digest = run.sha256_file(path)
        require(expected is None or digest == expected, 'changed frozen input: ' + name)
        inputs[name] = digest
    for filename in ('registration.json', 'report.json', 'collect.complete.json', 'audit.complete.json'):
        bind(SOURCE+'/'+filename)
    cases_file = registration['config']['source']+'/cases.json'
    bind(cases_file, registration['inputs'][cases_file])
    cases = run.check_seal(run.read_json(ROOT/cases_file))
    for j in registration['jobs']:
        if not j['comparison_arm'].startswith('raw_'):
            continue
        row = old.source.read_result(registration, src, j)
        prefix = SOURCE+'/episodes/'+j['job_id']+'/'
        for name, digest in row['files'].items():
            bind(prefix+name, digest)
        bind(prefix+'result.json')
        bind(j['model']['path'], j['model']['sha256'])
        bind(j['plan']['native_file'], j['plan']['config']['native_sha256'])
        for name in j['case']['files'].values():
            bind(name, cases['files'][name])
        jobs.append(dict(j, result_sha256=inputs[prefix+'result.json']))
    require(len(jobs) == 96 and len(old.sample_jobs(jobs)) == 12, 'frozen scope changed')
    for directory in ('experiments', 'scripts', 'lns2_selector'):
        for path in (ROOT/directory).rglob('*.py'):
            bind(path.relative_to(ROOT).as_posix())
    for path in (ROOT/'artifacts/initlns-closed-loop-controller-v2').rglob('*.json'):
        name = path.relative_to(ROOT).as_posix()
        bind(name, registration['inputs'][name])
    for name in ('tests/evaluation/test_sa_shared_features.py', 'docs/SA_SHARED_FEATURE_PROTOCOL_ZH.md'):
        bind(name)
    body = dict(schema='lns2.sa.shared_features.audit.v1', source_binding=registration['binding'],
                inputs=inputs, jobs=jobs, workers=20, component_workers=1, repeats=3,
                no_training=True, no_ttf=True, no_default_change=True)
    body['binding'] = run.json_fingerprint(body)
    with old.recovery.strict_lock(out, body['binding'], 'shared-prepare'):
        run.once(out/'registration.json', run.sealed(body))
    return dict(episodes=96, component_states=36, native_prefix_steps=36, inputs=len(inputs))


def registered(output):
    out = ROOT/output
    r = run.check_seal(run.read_json(out/'registration.json'))
    require(r['binding'] == run.json_fingerprint({k: v for k, v in r.items() if k not in ('binding', 'integrity')}),
            'registration binding changed')
    for name, digest in r['inputs'].items():
        require(run.sha256_file(run.contained_file(ROOT, name, field='audit input')) == digest,
                'changed audit input: ' + name)
    return r, out


def execute(phase, output):
    r, out = registered(output)
    jobs = r['jobs'] if phase == 'verify' else old.sample_jobs(r['jobs'])
    if phase in ('benchmark', 'native'):
        old.source.check_complete(r, out, 'verify', r['jobs'])
    jobs = [dict(j, parent_pid=os.getpid(), timing_binding=r['source_binding'], mode=phase, sample_index=i)
            for i, j in enumerate(jobs)]
    with old.recovery.strict_lock(out, r['binding'], 'shared-'+phase):
        require(not (out/(phase+'.complete.json')).exists(), 'phase already complete')
        pending = []
        for j in jobs:
            path = out/phase/(j['job_id']+'.json')
            if path.exists():
                receipt = run.check_seal(run.read_json(path))
                require(receipt['status'] == 'ok' and receipt['binding'] == r['binding'], 'failed receipt; inspect')
            else:
                pending.append(j)
        run.write_json(out/'run_status.json', dict(status='running', phase=phase, pending=len(pending)))
        try:
            old.source.execute(r, out, pending, native_worker if phase == 'native' else worker, phase,
                               1 if phase in ('profile', 'benchmark') else r['workers'], 900.)
            files = {j['job_id']: run.sha256_file(out/phase/(j['job_id']+'.json')) for j in jobs}
            run.once(out/(phase+'.complete.json'), run.sealed(dict(binding=r['binding'], jobs=len(jobs), files=files)))
            run.write_json(out/'run_status.json', dict(status='phase_complete', phase=phase, jobs=len(jobs)))
        except BaseException as exc:
            run.write_json(out/'run_status.json', dict(status='interrupted_or_error', phase=phase, error=str(exc)))
            raise
    return dict(phase=phase, jobs=len(jobs), no_ttf=True)


def report(output):
    r, out = registered(output)
    results = {}
    for phase in ('profile', 'verify', 'benchmark', 'native'):
        jobs = r['jobs'] if phase == 'verify' else old.sample_jobs(r['jobs'])
        old.source.check_complete(r, out, phase, jobs)
        results[phase] = [run.check_seal(run.read_json(out/phase/(j['job_id']+'.json'))) for j in jobs]
    states = []
    for receipt in results['benchmark']:
        for s in receipt['samples']:
            times = {k: statistics.median(o['seconds'] for o in s['observations'] if o['mode'] == k)
                     for k in ('reference', 'shared')}
            states.append(dict(job_id=receipt['job_id'], map_id=receipt['map_id'], decision=s['decision'], **times))
    a, b = (statistics.mean(s[k] for s in states) for k in ('reference', 'shared'))
    body = dict(schema='lns2.sa.shared_features.report.v1', binding=r['binding'], episodes=96,
                decisions=sum(x['decisions'] for x in results['verify']),
                candidates=sum(x['candidates'] for x in results['verify']),
                native_prefix_steps=sum(x['steps'] for x in results['native']), states=states,
                reference_mean_seconds=a, shared_mean_seconds=b, reduction=1-b/a,
                faster_states=sum(s['shared'] < s['reference'] for s in states),
                no_training=True, no_ttf=True, no_default_change=True,
                measured_scope='archived proposals; topology + GBDT features/scoring + actor; excludes proposal and PP')
    run.once(out/'report.json', run.sealed(body))
    run.write_json(out/'run_status.json', dict(status='complete', binding=r['binding']))
    return {k: v for k, v in body.items() if k != 'states'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare', 'profile', 'verify', 'benchmark', 'native', 'report'))
    p.add_argument('--output', default=OUTPUT)
    args = p.parse_args()
    value = prepare(args.output) if args.phase == 'prepare' else report(args.output) if args.phase == 'report' else execute(args.phase, args.output)
    print(json.dumps(value, indent=2), flush=True)


if __name__ == '__main__':
    main()
