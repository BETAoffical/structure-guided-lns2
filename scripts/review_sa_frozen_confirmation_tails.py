"""Posthoc, read-only failure review on paired completed confirmation traces."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_onpolicy as run
from experiments.sa_expanded_trace_diagnostic import analyze, first_divergence

SOURCE = ROOT/'build/sa-frozen-confirmation-ttf-v1'
OUTPUT = ROOT/'build/sa-frozen-confirmation-tail-review-v1'
ARMS = {'official', 'official_sa', 'parent', 'completion'}


def selected_pairs(rows):
    groups = {}
    for row in rows:
        g = groups.setdefault(row['pair_id'], {})
        run.require(row['comparison_arm'] not in g, 'duplicate arm')
        g[row['comparison_arm']] = row
    run.require(all(set(g) == ARMS for g in groups.values()), 'incomplete pairing')
    return {k:g for k,g in sorted(groups.items()) if any(not r['success_within_budget'] for r in g.values())}


def worker(item):
    row, binding = item
    folder = SOURCE/'episodes'/row['job_id']
    run.require(run.read_json(folder/'result.json') == row, 'source result changed')
    for name, sha in row['files'].items():
        run.require(run.sha256_file(run.contained_file(folder,name,field='trace')) == sha, 'changed trace')
    proof = run.check_seal(run.read_json(SOURCE/'audit'/(row['job_id']+'.json')))
    run.require(proof['result_sha256'] == run.sha256_file(folder/'result.json'), 'stale audit')
    with gzip.open(folder/'trace.jsonl.gz','rt',encoding='utf-8') as f:
        record = analyze(run.read_json(folder/'initial.json'),run.read_json(folder/'final.json'),
                         (json.loads(line) for line in f),row)
    record['binding'] = binding
    run.once(OUTPUT/'episodes'/(row['job_id']+'.json'),run.sealed(record))
    return record


def main():
    run.require(not OUTPUT.exists(), 'review exists; inspect rather than overwrite')
    source = run.check_seal(run.read_json(SOURCE/'report.json'))
    run.require(run.read_json(SOURCE/'run_status.json')['status'] == 'complete', 'source unfinished')
    groups = selected_pairs(source['episodes'])
    rows = [r for g in groups.values() for _,r in sorted(g.items())]
    inputs = {str(p.relative_to(ROOT).as_posix()):run.sha256_file(p) for p in (
        SOURCE/'report.json', SOURCE/'audit.complete.json', Path(__file__).resolve(),
        ROOT/'experiments/sa_expanded_trace_diagnostic.py',
        ROOT/'tests/evaluation/test_sa_frozen_confirmation_tail_review.py')}
    body = dict(schema='lns2.sa_frozen_confirmation_tail_review.v1',inputs=inputs,
        pair_ids=list(groups),episodes=len(rows),workers=20,selection='any_arm_failed_all_four_arms_retained',
        retrospective=True,no_solver_calls=True,no_training=True,not_a_causal_intervention=True)
    binding = run.json_fingerprint(body)
    run.once(OUTPUT/'registration.json',run.sealed(dict(body,binding=binding)))
    results = []
    with ProcessPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(worker,(row,binding)) for row in rows]
        for f in as_completed(futures):
            results.append(f.result())
            run.write_json(OUTPUT/'progress.json',dict(completed=len(results),total=len(rows),status='running'))
    paired = {}
    for r in results:
        paired.setdefault(r['pair_id'],{})[r['arm']] = r
    report = dict(binding=binding,source_report_sha256=inputs['build/sa-frozen-confirmation-ttf-v1/report.json'],
        episodes=len(results),pairs=len(groups),retrospective=True,no_solver_calls=True,
        not_all_episode_frequency_estimate=True,not_a_causal_intervention=True,
        records=[{k:v for k,v in r.items() if k != 'decision_signatures'} for r in sorted(results,key=lambda x:x['job_id'])],
        actor_divergences={p:first_divergence(g['parent'],g['completion']) for p,g in sorted(paired.items())},
        receipts={r['job_id']:run.sha256_file(OUTPUT/'episodes'/(r['job_id']+'.json')) for r in results})
    run.once(OUTPUT/'report.json',run.sealed(report))
    run.write_json(OUTPUT/'progress.json',dict(completed=len(results),total=len(rows),status='complete'))
    print(json.dumps(dict(episodes=len(results),pairs=len(groups),report_sha256=run.sha256_file(OUTPUT/'report.json'))))


if __name__ == '__main__':
    main()
