"""Read retained traces only; no solver, learned predictions or new outcomes."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from experiments.closed_loop_trace_storage import apply_state_delta
from experiments.repair_collection import state_fingerprint

SOURCE = ROOT / 'build/path-quality-pressure-deadline-recovery-v1'
REVIEW = SOURCE / 'review-20260909'
OUT = ROOT / 'build/initlns-preentry-trace-audit-v1'


def native_path(path):
    return '\\\\?\\' + str(path.resolve()) if os.name == 'nt' else str(path)


def read(path):
    with open(native_path(path), encoding='utf-8') as stream:
        return json.load(stream)


def sha(path):
    digest = hashlib.sha256()
    with open(native_path(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def edges(state):
    return {tuple(sorted(edge)) for edge in state['conflict_edges']}


def exit_availability(run, exit_record):
    if not run or exit_record is None:
        return None
    cid = exit_record['selection']['candidate_id']
    appearances = []
    for row in run:
        pool = row['selection']['alternatives']
        match = next((p for p in pool if p['candidate_id'] == cid), None)
        if match:
            scores = [p['score'] for p in pool] + [row['selection']['score']]
            appearances.append(dict(step=row['step'], rank=1+sum(s>match['score'] for s in scores)))
    return dict(unselected_appearances=len(appearances), first=appearances[:1],
                best_rank=min((r['rank'] for r in appearances), default=None))


def pool_record(event):
    pool = event['controller']['candidate_pool']
    assert len({p['candidate_id'] for p in pool}) == len(pool)
    selected = event['controller']['selected_candidate_id']
    item = next(p for p in pool if p['candidate_id'] == selected)
    assert sorted(item['agents']) == sorted(event['metrics']['neighborhood'])
    return dict(candidate_id=selected, size=len(item['agents']),
                families=item['selection_families'], score=item['score'],
                rank=1 + sum(p['score'] > item['score'] for p in pool),
                pool_count=len(pool),
                alternatives=[dict(candidate_id=p['candidate_id'], agents=p['agents'],
                    score=p['score'], families=p['selection_families']) for p in pool
                    if p['candidate_id'] != selected])


def audit(case):
    old_path = REVIEW / 'transition-audit/episodes' / (case['source_job_id'] + '.json')
    old = read(old_path)
    trace = ROOT / case['source_trace']
    assert sha(trace) == case['source_trace_sha256'] == old['trace_sha256']
    records = []
    fingerprint = None
    with gzip.open(native_path(trace), 'rt', encoding='utf-8') as stream:
        for line in stream:
            event = json.loads(line)
            if event['event'] == 'initial':
                blob = SOURCE / 'episodes' / case['source_job_id'] / 'first_phase' / event['state_blob']
                with gzip.open(native_path(blob), 'rt', encoding='utf-8') as initial:
                    state = json.load(initial)
                fingerprint = state_fingerprint(state)
                assert fingerprint == event['state_fingerprint']
                initial_fingerprint = fingerprint
                continue
            if event['event'] == 'finish':
                continue
            assert event['event'] == 'transition'
            assert event['before_fingerprint'] == fingerprint
            before = state
            state = apply_state_delta(state, event['state_delta'])
            fingerprint = state_fingerprint(state)
            assert fingerprint == event['after_fingerprint']
            m = event['metrics']
            assert m['action_valid'] and m['conflicts_before'] == len(edges(before))
            assert m['conflicts_after'] == len(edges(state))
            old_agents = {a['id']: a for a in before['agents']}
            changed = [a['id'] for a in state['agents'] if a['path'] != old_agents[a['id']]['path']]
            assert set(changed) <= set(m['neighborhood'])
            same = not changed and edges(before) == edges(state) and before['sum_of_costs'] == state['sum_of_costs']
            if m['pp_rolled_back']:
                assert same
            step = event['decision_index'] + 1
            assert step == len(records) + 1
            records.append(dict(step=step, before=m['conflicts_before'], after=m['conflicts_after'],
                same=same, rollback=m['pp_rolled_back'], reason=m['pp_failure_reason'],
                changed_agents=changed, added_edges=sorted(edges(state)-edges(before)),
                removed_edges=sorted(edges(before)-edges(state)),
                soc_delta=state['sum_of_costs']-before['sum_of_costs'],
                pp_seconds=m['pp_replan_seconds'], order=m['repair_order'],
                selection=pool_record(event)))
    assert len(records) == old['steps'] and state['num_of_colliding_pairs'] == old['final_conflicts']
    # A run ends at the first changed path, not necessarily at a conflict decrease.
    runs = []
    index = 0
    while index < len(records):
        if not records[index]['same']:
            index += 1
            continue
        end = index
        while end < len(records) and records[end]['same']:
            end += 1
        runs.append((index, end))
        index = end
    start, end = max(runs, key=lambda r: (r[1]-r[0], -r[0]), default=(0, 0))
    run = records[start:end]
    assert len(run) == old['max_same_state_run']
    counts = Counter(r['selection']['candidate_id'] for r in run)
    selected_ids = set(counts)
    available = {p['candidate_id'] for r in run for p in r['selection']['alternatives']} | selected_ids
    entry = records[start-1] if start else None
    exit_record = records[end] if run and end < len(records) else None
    drops = [r for r in records[end:] if r['after'] < r['before']] if run else []
    result = dict(case_id=case['case_id'], controller=case['controller'], role=case['role'],
        task_id=case['task_id'], solver_seed=case['solver_seed'], success=old['success'],
        source_trace=case['source_trace'], source_sha256=sha(trace),
        prior_audit_sha256=sha(old_path), initial_fingerprint=initial_fingerprint,
        steps=len(records), final_conflicts=old['final_conflicts'],
        longest_unchanged=dict(first_step=start+1 if run else None, last_step=end if run else None,
            attempts=len(run), pp_seconds=sum(r['pp_seconds'] for r in run),
            conflicts=run[0]['before'] if run else None,
            selected_counts=dict(counts), available_candidates=len(available),
            unselected_available_candidates=len(available-selected_ids),
            unique_orders=len({tuple(r['order']) for r in run}),
            failure_reasons=dict(Counter(r['reason'] for r in run))),
        entry_transition=entry, exit_transition=exit_record,
        exit_candidate_prior_availability=exit_availability(run, exit_record),
        exit_candidate_previously_tried=bool(exit_record and exit_record['selection']['candidate_id'] in selected_ids),
        exit_order_previously_tried=bool(exit_record and any(r['selection']['candidate_id']==exit_record['selection']['candidate_id']
            and r['order']==exit_record['order'] for r in run)),
        next_strict_drop=drops[0]['step'] if drops else None,
        last_transitions_before_entry=records[max(0,start-5):start],
        last_transitions=records[-3:],
        interpretation='Post-hoc six-case trace association; unselected candidate outcomes unknown.')
    return result


if __name__ == '__main__':
    plan = read(REVIEW / 'continuation-diagnostic-v2/plan.json')
    OUT.mkdir(exist_ok=True)
    # Six independent input traces: extra idle processes would not add throughput.
    with ProcessPoolExecutor(max_workers=min(20, len(plan['cases']))) as executor:
        results = list(executor.map(audit, plan['cases']))
    report = dict(schema='lns2.preentry_trace_audit.v1', cases=results,
        script_sha256=sha(Path(__file__)), errors=0, solver_runs=0,
        caveat='No counterfactual outcomes collected. No causal or TTF improvement claim.')
    target = OUT / 'report.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(target)
    for r in results:
        print(json.dumps(dict(case_id=r['case_id'], success=r['success'], steps=r['steps'],
            unchanged=r['longest_unchanged']['attempts'],
            exit_candidate_prior_availability=r['exit_candidate_prior_availability']), ensure_ascii=False), flush=True)
