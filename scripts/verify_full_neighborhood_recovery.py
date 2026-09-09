"""Independent saved-result checks and descriptive failure attribution; no solving."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._common import read_json, write_json, write_jsonl
from experiments.full_neighborhood_recovery import METHODS, check_paths, mechanism_gate, scientific_signature
from experiments.full_recovery_collection import check_output, failure_label, load, output_file
from experiments.local_path_compatibility import ROOT, contained, digest, sha256_file
from experiments.native_path_compatibility import check_seal, paths_of, seal


def verify(output):
    m=load(output)
    report=read_json(output/'report.json')
    check_seal(report)
    if not report['complete'] or not report['parity_passed'] or report['fingerprint']!=m['content_sha256']:
        raise ValueError('complete parity and mechanism evidence required')
    cases={c['case_id']:c for c in m['cases']}
    states={key:read_json(contained(case['state_file'])) for key,case in cases.items()}
    transitions={key:read_json(contained(case['historical_transition_file'])) for key,case in cases.items()}
    rows, grouped, per_case=[],defaultdict(list),defaultdict(lambda:defaultdict(Counter))
    conditions=[]
    checks=Counter()
    for job in m['jobs']:
        path=output_file(output,job)
        row=read_json(path)
        check_output(m,job,row)
        if report['result_hashes'][path.relative_to(ROOT).as_posix()]!=sha256_file(path):
            raise ValueError('report-result SHA mismatch')
        if job['method']=='parity':
            if row['status']!='pass' or not row['checks'] or not all(row['checks'].values()):
                raise ValueError('parity failed')
            checks['parity']+=1
            continue
        rows.append(row)
        grouped[(job['case_id'],job['seed'])].append(row)
        if row['status'] not in ('ok','unknown'):
            raise ValueError('unexplained worker error')
        if row['status']=='ok':
            state=states[job['case_id']]
            order=transitions[job['case_id']]['metrics']['repair_order']
            if check_paths(state,row['final']['paths'],order)!=row['path_check'] or not row['native_paths_verified']:
                raise ValueError('final-path verification mismatch')
            checks['final_paths']+=1
            if not row['recovered'] and row['final']!=row['base']:
                raise ValueError('unsuccessful recovery failed to preserve base result')
            if row['base']['status']=='rolled_back' and row['base']['paths']!=paths_of(state):
                raise ValueError('ordinary rollback changed source paths')
            if row['recovered']:
                if row['final']['conflicts']>=row['base']['conflicts']:
                    raise ValueError('claimed recovery does not improve conflicts')
                checks['recoveries']+=1
            for attempt in row['attempts']:
                option, branch=attempt['option'],attempt['result']
                if option:
                    if option['blocker'] not in order or option['victim'] not in order:
                        raise ValueError('feedback changes external agent')
                    if option['prefix_length']!=order.index(option['blocker']):
                        raise ValueError('backtrack prefix mismatch')
                if branch['status']=='accepted':
                    check_paths(state,branch['paths'],order)
                    if option and len(branch['paths'][option['blocker']])-1>option['cap']:
                        raise ValueError('cost cap violation')
                checks['recovery_attempts']+=1
        per_case[job['case_id']][job['method']][failure_label(row)]+=1
        case=cases[job['case_id']]
        feedback=row.get('feedback') or {}
        records=row.get('base',{}).get('records',[])
        order=transitions[job['case_id']]['metrics']['repair_order']
        positions={aid:i for i,aid in enumerate(order)}
        internal_counts=[]
        for record in records:
            victim=record['agent']
            blockers=[e['right'] if e['left']==victim else e['left'] for e in record['incident_events']]
            internal_counts.append(sum(b in positions and positions[b]<positions[victim] for b in blockers))
        conditions.append(dict(job=job,role=case['role'],map_id=case['map_id'],controller=case['controller'],
            category=failure_label(row),initial_conflicts=states[job['case_id']]['num_of_colliding_pairs'],
            base_conflicts=row.get('base',{}).get('conflicts'),final_conflicts=row.get('final',{}).get('conflicts'),
            base_status=row.get('base',{}).get('status'),triggered=row.get('triggered',False),
            recovered=row.get('recovered',False),feedback_candidates=len(feedback.get('candidates',[])),
            relaxed_searches=len(feedback.get('evidence',[])),external_blockers=feedback.get('external_blockers',[]),
            recorded_internal_events=sum(internal_counts),
            earlier_internal_events_before_failed_insertion=sum(internal_counts[:-1])
                if row.get('base',{}).get('status')=='rolled_back' else None,
            attempt_statuses=dict(Counter(a['result']['status'] for a in row.get('attempts',[]))),
            low_level_calls=row.get('low_level_calls'),expanded=row.get('expanded'),
            diagnostic_seconds=row.get('diagnostic_seconds')))
    for key, group in grouped.items():
        if len(group)!=len(METHODS) or {r['job']['method'] for r in group}!=set(METHODS):
            raise ValueError('missing paired method')
        valid=[r for r in group if r['status']=='ok' and not r['budget_exhausted']]
        if len({digest(scientific_signature(r['base'])) for r in valid})>1:
            raise ValueError('paired ordinary baselines differ: '+str(key))
        checks['paired_baselines']+=1
    gate=mechanism_gate(rows,m['cases'],len(m['cases'])*len(m['config']['pp_seeds'])*len(METHODS))
    if len(m['cases'])==m['config']['maximum_states'] and gate!=report['gate']:
        raise ValueError('gate reconstruction differs')
    if report['timing_allowed']:
        raise ValueError('mechanism report must not enable timing without native transaction integration')
    descriptions=[]
    for key,methods in per_case.items():
        case=cases[key]
        descriptions.append(dict(case_id=key,role=case['role'],map_id=case['map_id'],controller=case['controller'],
            methods={name:dict(counts) for name,counts in methods.items()}))
    write_jsonl(output/'case_failure_summary.jsonl',descriptions)
    write_jsonl(output/'condition_diagnostics.jsonl',conditions)
    value=seal(dict(schema='lns2.full_recovery_verification.v1',passed=True,fingerprint=m['content_sha256'],
        counts=dict(checks),gate=report['gate'],timing_allowed=False,
        registered_hashes=len(m['files']),peak_worker_rss_mib=max(r.get('max_rss_kib',0) for r in rows)/1024,
        report_sha256=sha256_file(output/'report.json'),verifier_sha256=sha256_file(Path(__file__))))
    write_json(output/'verification.json',value)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='build/initlns-full-neighborhood-recovery-v1-final')
    args=parser.parse_args()
    print(verify(contained(args.output)))
