"""Reconstruct retained proposal grids; never repair a new candidate."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from scripts.audit_preentry_trace import REVIEW, native_path, read, sha
from experiments.neighborhood_candidates import candidate_id, select_representative_neighborhood_groups
from experiments.repair_collection import state_fingerprint, select_seed_agents
from lns2_selector.runtime.online_selection import proposal_random_seeds

OUT = ROOT / 'build/initlns-recovery-provenance-audit-v1'
PREVIOUS = ROOT / 'build/initlns-preentry-trace-audit-v1/report.json'
FIELDS = ('agents', 'candidate_id', 'proposal_count_by_family', 'proposal_seeds',
          'seed_agents', 'selection_families', 'selection_rank_by_family')


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    temp.replace(path)


def cases():
    evidence = read(ROOT / 'artifacts/initlns-preentry-trace-audit-v1/evidence.json')
    assert sha(PREVIOUS) == evidence['report_sha256']
    prior = read(PREVIOUS)
    selected = [r for r in prior['cases'] if r['role'] != 'fast_control' and r['exit_transition']]
    assert len(selected) == 3
    return selected


def reconstruct_groups(payload, specs, seeds):
    assert not payload['invalid_indices'] and not payload.get('deadline_indices')
    assert payload['proposal_count'] == len(specs)
    covered = []
    groups = []
    for agents, indices in payload['rows']:
        covered.extend(indices)
        groups.append(dict(agents=list(agents), sources=[dict(
            family=f'{specs[i][1]}:{specs[i][2]}', seed_agent=specs[i][0], proposal_seed=seeds[i])
            for i in indices]))
    assert sorted(covered) == list(range(len(specs)))
    assert len(groups) == payload['unique_neighborhood_count']
    return groups


def child(case_id):
    detail = next(r for r in cases() if r['case_id'] == case_id)
    plan = read(REVIEW / 'continuation-diagnostic-v2/plan.json')
    case = next(r for r in plan['cases'] if r['case_id'] == case_id)
    source = read(ROOT / case['state_file'])
    assert case['decision_index']+1 == detail['longest_unchanged']['first_step']
    conf = read(ROOT / 'configs/path_quality_pressure_deadline_recovery_v1.json')
    native = ROOT / conf['native_file']
    bound = [case[k] for k in ('state_file','prefix_file','map_file','scenario_file','source_trace')]
    bound += ['experiments/neighborhood_candidates.py', 'experiments/repair_collection.py',
              'lns2_selector/runtime/online_selection.py', conf['native_file']]
    for path in bound:
        assert sha(ROOT/path) == plan['files'][path], 'input changed: '+path
    assert sha(native) == conf['native_sha256']
    sys.path.insert(0, str(native.parent))
    import lns2_env
    assert Path(lns2_env.__file__).resolve() == native.resolve()
    env = lns2_env.LNS2RepairEnv(str(ROOT/case['map_file']), str(ROOT/case['scenario_file']),
        case['agent_count'], time_limit=600, neighborhood_size=8, destroy_strategy='Adaptive',
        replan_algorithm='PP', use_sipp=True, max_repair_iterations=0, screen=0, context=source.get('context'))
    state = env.reset(case['solver_seed'])
    prefix = [json.loads(line) for line in (ROOT/case['prefix_file']).read_text().splitlines()]
    for row in prefix:
        assert state_fingerprint(state) == row['before_fingerprint']
        result = env.step_with_time_limit(row['action'], row['original_pp_time_limit_seconds'])
        state = result['observation']
        assert state_fingerprint(state) == row['after_fingerprint']
        assert result['metrics']['repair_order'] == row['original_repair_order']
    anchor = state_fingerprint(state)
    assert anchor == case['full_state_fingerprint']
    assert len(prefix) == case['prefix_actions']
    exit_step = detail['exit_transition']['step']
    target = detail['exit_transition']['selection']['candidate_id']
    rows = []; request_count = 0; source_seeds = None
    with gzip.open(native_path(ROOT/case['source_trace']), 'rt', encoding='utf-8') as stream:
        for line in stream:
            event = json.loads(line)
            if event['event'] != 'transition':
                continue
            step = event['decision_index']+1
            if step < detail['longest_unchanged']['first_step']:
                continue
            if step > exit_step:
                break
            meta = event['controller']['proposal']
            selected_seeds = select_seed_agents(state, 4, state_hash=event['before_fingerprint'])
            assert selected_seeds == meta['seed_agents']
            specs = [(s, h, size, trial) for s in selected_seeds
                     for h in ('target','collision','random') for size in (4,8,16) for trial in range(8)]
            seeds = proposal_random_seeds(case['task_id'], case['solver_seed'], event['before_fingerprint'],
                                          event['decision_index'], specs)
            groups = reconstruct_groups(env.propose_seed_grid_grouped(selected_seeds,
                ['target','collision','random'], [4,8,16], seeds, 8), specs, seeds)
            assert len(specs) == meta['proposal_count']
            assert len(groups) == meta['unique_neighborhood_count']
            retained = select_representative_neighborhood_groups(groups, 2)
            original = [p for p in event['controller']['candidate_pool']
                        if any(f.split(':')[0] in ('target','collision','random') for f in p['selection_families'])]
            signature = lambda pool: sorted([{k:p[k] for k in FIELDS} for p in pool], key=lambda p:p['candidate_id'])
            assert signature(retained) == signature(original), f'retained pool mismatch: {step}'
            assert state_fingerprint(env.get_state()) == anchor, 'proposal changed environment'
            match = [g for g in groups if candidate_id(g['agents']) == target]
            is_retained = any(p['candidate_id'] == target for p in retained)
            rows.append(dict(step=step, seed_agents=selected_seeds, raw_present=bool(match),
                retained=is_retained, sources=match[0]['sources'] if match else [],
                raw_count=len(groups), requested=len(specs)))
            request_count += len(specs)
            if step == exit_step:
                witness = next(p for p in original if p['candidate_id'] == target)
                source_seeds = witness['seed_agents']
            if len(rows) % 200 == 0:
                print(json.dumps(dict(case_id=case_id, proposal_grids=len(rows))), flush=True)
    assert rows and rows[-1]['step'] == exit_step and rows[-1]['raw_present'] and rows[-1]['retained']
    history = rows[:-1]
    report = dict(case_id=case_id, status='ok', source_trace_sha256=case['source_trace_sha256'],
        target_candidate_id=target, witness=witness, prefix_repair_replays=len(prefix),
        new_candidate_repairs=0, proposal_grids=len(rows), proposal_requests=request_count,
        previous_raw_hits=sum(r['raw_present'] for r in history),
        previous_retained_hits=sum(r['retained'] for r in history),
        previous_pruned_hits=sum(r['raw_present'] and not r['retained'] for r in history),
        earliest_raw_step=next(r['step'] for r in rows if r['raw_present']),
        source_seed_coverage={str(s):sum(s in r['seed_agents'] for r in history) for s in source_seeds},
        archived_event_outcome=dict(before=detail['exit_transition']['before'],
            after=detail['exit_transition']['after'], episode_success=detail['success']),
        input_sha256={p:plan['files'][p] for p in bound},
        caveat='Generator-only replay at identical paths, with historical per-step request seeds; counters not replayed. All retained pool provenance must match. No witness repair repeated.',
        rows=rows)
    write(OUT/(case_id+'.json'), report)


def main():
    OUT.mkdir(exist_ok=True)
    jobs = cases()
    active = []
    try:
        for job in jobs:
            path = OUT/(job['case_id']+'.json')
            if path.exists():
                raise RuntimeError('Existing diagnostic output: refuse to overwrite')
        for job in jobs:
            log = (OUT/(job['case_id']+'.log')).open('w')
            proc = subprocess.Popen([sys.executable,'-B',__file__,'_child',job['case_id']],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            active.append((proc, log, time.monotonic(), job))
        while active:
            for proc, log, start, job in list(active):
                if time.monotonic()-start > 180:
                    raise TimeoutError('Proposal replay fuse: '+job['case_id'])
                if proc.poll() is None:
                    continue
                log.close(); active.remove((proc,log,start,job))
                if proc.returncode:
                    raise RuntimeError('Replay failed: '+job['case_id'])
                print(json.dumps(dict(completed=job['case_id'])), flush=True)
            time.sleep(0.2)
    finally:
        for proc, log, _, _ in active:
            if proc.poll() is None:
                proc.kill()
            proc.wait(); log.close()
    summaries = [{k:v for k,v in read(OUT/(j['case_id']+'.json')).items() if k!='rows'} for j in jobs]
    write(OUT/'report.json', dict(schema='lns2.recovery_provenance.v1', cases=summaries,
        script_sha256=sha(Path(__file__)), previous_report_sha256=sha(PREVIOUS),
        errors=0, new_candidate_repairs=0, timing_experiment=False))
    print(json.dumps([{k:r[k] for k in ('case_id','proposal_grids','previous_raw_hits',
        'previous_pruned_hits','earliest_raw_step','source_seed_coverage')} for r in summaries]), flush=True)


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='_child':
        child(sys.argv[2])
    else:
        main()
