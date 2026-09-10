"""Post-hoc failure descriptions; cannot change the preregistered opportunity gate."""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True

from scripts.diagnose_single_release_opportunity import (
    Engine, check_seal, digest, edge_set, load, read_json, read_result,
    require, seal, sha256_file, signature, write_json,
)

_DATA = None


def init(plan, output):
    global _DATA
    contexts = {}
    for c in plan['conditions']:
        s = read_json(ROOT/c['case']['state_file'])
        base = read_json(ROOT/c['source_result'])['base']
        contexts[c['id']] = (c, s, signature(base))
    _DATA = plan, Path(output), contexts


def describe(job):
    plan, out, contexts = _DATA
    row = read_result(plan, out, job); c, state, base = contexts[job['condition']]
    count = Counter(total=1)
    if row['status'] != 'ok':
        count['unknown_or_not_found'] += 1
        return job['condition'], dict(count)
    with gzip.open(out/'raw'/(job['id']+'.json.gz'), 'rt', encoding='utf-8') as f:
        raw = json.load(f)
    check_seal(raw); raw = raw['raw']
    current = [(r['agent'], r['path']) for r in raw['diagnostics']]
    previous = [(r['agent'], r['path']) for r in base['diagnostics']]
    count['same_attempted_paths_and_stop' if current==previous else 'different_attempted_path_or_stop'] += 1
    base_count = len(edge_set(dict(enumerate(base['paths']))))
    if raw['rolled_back']:
        count['rolled_back'] += 1
        original = {a['id']: a['path'] for a in state['agents']}
        visible = {i: p for i, p in original.items() if i not in c['order']}
        visible.update(current)
        pairs = edge_set(visible, [i for i, _ in current])
        require(len(pairs)==raw['attempted_pairs'], 'posthoc conflict ledger')
        new_pairs = pairs-edge_set(original, c['order'])
        count['rollback_with_new_pair_touching_released_agent'] += any(job['released'] in p for p in new_pairs)
    else:
        difference = row['path_check']['conflicts']-base_count
        count['accepted_better' if difference<0 else 'accepted_worse' if difference>0 else 'accepted_tie'] += 1
    return job['condition'], dict(count)


def main():
    plan, out = load(); report = read_json(out/'report.json'); check_seal(report)
    require(report['plan']==plan['content_sha256'] and report['total']==len(plan['jobs']), 'complete primary audit required')
    for j in plan['jobs']:
        require(sha256_file(out/'results'/(j['id']+'.json'))==report['result_sha256'][j['id']], 'primary result changed')
    counts = defaultdict(Counter)
    with mp.get_context('spawn').Pool(plan['config']['workers'], initializer=init, initargs=(plan, str(out))) as pool:
        for condition, values in pool.imap(describe, plan['jobs'], chunksize=4):
            counts[condition].update(values)
    # Deterministic, outcome-blind fresh-process-context replay, one action per condition.
    replay = []
    for c in plan['conditions']:
        job = min((j for j in plan['jobs'] if j['condition']==c['id']), key=lambda j: j['id'])
        engine = Engine(plan)
        raw, _ = engine.release(c, job['released'])
        with gzip.open(out/'raw'/(job['id']+'.json.gz'), 'rt', encoding='utf-8') as f:
            saved = json.load(f)['raw']
        require(raw['status']==saved['status']=='ok', 'fresh context replay unknown')
        scientific = lambda value: {k:v for k,v in signature(value).items() if k!='paths'}
        # The compact record has only the final path digest, but full per-search paths.
        require(scientific(raw)==scientific(dict(saved, paths=[])), 'fresh-context scientific replay')
        require(digest(raw['paths'])==saved['paths_sha256'], 'fresh-context final paths')
        replay.append(dict(job=job['id'], matched=True))
    result = seal(dict(schema='lns2.single_release_posthoc.v1', plan=plan['content_sha256'],
                       primary_report_sha256=sha256_file(out/'report.json'),
                       implementation_sha256=sha256_file(Path(__file__)),
                       conditions={k:dict(v) for k,v in counts.items()}, replay=replay,
                       changes_primary_gate=False, scope='descriptive_development_failure_audit'))
    write_json(out/'failure-analysis.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
