"""Independent real-path ledgers and the scheduled two-round failure review."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.dont_write_bytecode=True
from experiments._common import read_json, write_json
from experiments.state_analysis import reconstruct_conflicts
from experiments.native_path_compatibility import edge_set, paths_of, modules
from experiments.full_neighborhood_recovery import check_paths
from scripts.diagnose_reservation_feedback import OUT, METHODS, load, read_result, feedback_ranking, gate
from scripts.diagnose_reservation_mediation import require, sha256_file


def canonical_role(role):
    roles={'failed_long':'failed_tail','failed_long_unchanged':'failed_tail',
           'successful_stall':'successful_tail','successful_long_unchanged':'successful_tail',
           'fast_control':'fast_control'}
    require(role in roles,'unknown case role: '+role)
    return roles[role]


def corrected_gate(rows):
    normalized=deepcopy(rows)
    for row in normalized:
        case=row['job']['case']
        if canonical_role(case['role'])=='failed_tail': case['role']='failed_long_unchanged'
    return gate(normalized)


def ledger(state,order,diagnostics):
    original={a['id']:a['path'] for a in state['agents']}
    visible={i:p for i,p in original.items() if i not in order}
    pairs=set(); snapshots=[]; records=[]; changed=[]
    for index,d in enumerate(diagnostics):
        aid=d['agent']; path=d['path']; a=state['agents'][aid]; cols=state['cols']
        require(aid==order[index] and d['status']=='path','invalid attempted order/status')
        require(path and path[0]==a['start'] and path[-1]==a['goal'],'attempt path endpoints')
        require(all(0<=x<len(state['obstacles']) and not state['obstacles'][x] for x in path),'attempt obstacle')
        require(all(abs(x//cols-y//cols)+abs(x%cols-y%cols)<=1 for x,y in zip(path,path[1:])),'attempt jump')
        visible[aid]=path
        events=[e for e in reconstruct_conflicts([dict(id=i,path=p) for i,p in visible.items()]) if aid in (e.left,e.right)]
        pairs.update((e.left,e.right) for e in events)
        snapshots.append(set(pairs))
        records.append(dict(agent=aid,incident_events=[dict(left=e.left,right=e.right,time=e.time,kind=e.kind) for e in events]))
        if path!=original[aid]: changed.append(aid)
    return dict(pairs=pairs,snapshots=snapshots,records=records,visible=visible,changed=changed)


def verify_branch(state,order,raw):
    result=ledger(state,order,raw['diagnostics']); original={a['id']:a['path'] for a in state['agents']}
    bound=len(edge_set(original,order)); crossed=[i for i,p in enumerate(result['snapshots']) if len(p)>bound]
    require(bool(crossed)==raw['rolled_back'],'rollback versus true occupancy')
    if crossed: require(crossed==[len(raw['diagnostics'])-1],'continued after acceptance bound')
    else: require(len(raw['diagnostics'])==len(order),'incomplete accepted sequence')
    expected=paths_of(state) if crossed else [result['visible'][i] for i in original]
    require(expected==raw['paths'],'committed paths not from complete guarded sequence')
    require(len(result['pairs'])==raw['attempted_pairs'],'attempt pair count')
    check_paths(state,raw['paths'],order)
    return result


def audit_job(job):
    r=read_result(job); state=read_json(ROOT/job['case']['state_file']); source=read_json(ROOT/job['source_result'])
    old=source['base']; base_diags=[dict(agent=x['agent'],**x['search']) for x in old['records']]
    bledger=ledger(state,job['order'],base_diags)
    require(feedback_ranking(dict(records=bledger['records']),job['order'])==job['selection']['ranked'],'feedback reconstruction')
    rows=[]
    for method in METHODS:
        b=r['branches'][method]
        row=dict(job=job['id'],case_id=job['case']['case_id'],map_id=job['case']['map_id'],
                 role=job['case']['role'],seed=job['seed'],method=method,applicable=b['applicable'],
                 baseline=r['base_conflicts'],final=b['final_conflicts'],recovered=b['recovered'])
        if b['applicable']:
            raw=b['raw']; al=verify_branch(state,job['order'],raw)
            expected=raw['paths'] if len(edge_set(dict(enumerate(raw['paths']))))<r['base_conflicts'] else old['paths']
            require(b['final_paths']==expected,'fallback mismatch')
            reference_diags=(r['branches']['fresh_retry']['raw']['diagnostics']
                             if method!='fresh_retry' else base_diags)
            reference_ledger=ledger(state,job['order'],reference_diags)
            common=min(len(al['snapshots']),len(reference_ledger['snapshots']))
            before=reference_ledger['snapshots'][common-1] if common else set()
            after=al['snapshots'][common-1] if common else set()
            chosen=b['chosen_agent']; removed=before-after; added=after-before
            same_attempt=(len(raw['diagnostics'])==len(base_diags) and
                          all(x['agent']==y['agent'] and x['path']==y['path'] for x,y in zip(raw['diagnostics'],base_diags)))
            row.update(chosen_agent=chosen,attempted_unchanged_from_old_seed=same_attempt,
                       common_prefix_comparator='fresh_retry' if method!='fresh_retry' else 'old_seed_ordinary',
                       attempt_same_as_comparator=(len(raw['diagnostics'])==len(reference_diags) and
                           all(x['agent']==y['agent'] and x['path']==y['path'] for x,y in zip(raw['diagnostics'],reference_diags))),
                       raw_conflicts=b['path_check']['conflicts'],rolled_back=raw['rolled_back'],
                       rollback_at=raw['diagnostics'][-1]['agent'] if raw['rolled_back'] else None,
                       removed_pairs_common_prefix=sorted(removed),added_pairs_common_prefix=sorted(added),
                       added_chosen_pairs=sorted(p for p in added if chosen in p),
                       added_other_pairs=sorted(p for p in added if chosen not in p),
                       removed_chosen_pairs=sorted(p for p in removed if chosen in p),
                       common_prefix_length=common,expanded=b['expanded'],generated=b['generated'],
                       classification='recovered' if b['recovered'] else 'rollback' if raw['rolled_back'] else 'accepted_without_improvement')
        else: row['classification']=b['reason']
        rows.append(row)
    return dict(rows=rows,base_expanded=sum(x['expanded'] for x in base_diags),
                base_generated=sum(x['generated'] for x in base_diags),
                directed_equals_random=job['selection']['directed']==job['selection']['random'],
                ranked_count=len(job['selection']['ranked']))


def witness_feedback_coverage():
    from scripts.diagnose_reservation_mediation import OUT as FIRST, OLD, load as first_load, read_result as first_result
    p=first_load(); state=read_json(OLD/'before.json'); rows=[]
    for job in p['jobs']:
        r=first_result('round1',job)
        action=read_json(OLD/f"{job['trial']:02d}-compressed11.json")['job']['action']
        seq=r['data']['sequences']['compressed11']; l=verify_branch(state,action['repair_order'],seq)
        ranked=feedback_ranking(dict(records=l['records']),action['repair_order'])
        ids=[x['agent'] for x in ranked]
        rows.append(dict(trial=job['trial'],witness_agent=p['released_agent'],
                         present=p['released_agent'] in ids,
                         rank=ids.index(p['released_agent'])+1 if p['released_agent'] in ids else None,
                         feedback_agents=ids))
    return rows


def verify_mediation_native():
    from scripts.diagnose_reservation_mediation import OUT as FIRST, OLD, load as first_load, read_result as first_result
    p=first_load(); state=read_json(OLD/'before.json'); native,_=modules(p['config'])
    env=native.LNS2RepairEnv(str(ROOT/p['case']['map_file']),str(ROOT/p['case']['scenario_file']),len(state['agents']),time_limit=60)
    count=0
    for j in p['jobs']:
        r=first_result('round2',j); raw=r['data']['result']
        order=read_json(OLD/f"{j['trial']:02d}-compressed11.json")['job']['action']['repair_order']
        verify_branch(state,order,raw)
        observed=env.reset_paths(raw['paths'],seed=0)
        require(paths_of(observed)==raw['paths'] and observed['num_of_colliding_pairs']==r['data']['path_check']['conflicts'],'mediation native verification')
        count+=1
    return count


def main():
    m=load(); report=read_json(OUT/'report.json')
    for j in m['jobs']:
        require(sha256_file(OUT/'results'/f"{j['id']}.json")==report['result_sha256'][j['id']],'report row SHA')
    saved=[read_result(j) for j in m['jobs']]
    corrected=deepcopy(report)
    corrected.update(schema='lns2.reservation_feedback_report.v1.1',
        gate=corrected_gate(saved),
        supersedes_report_sha256=sha256_file(OUT/'report.json'),
        correction='Reporting role aliases failed_long and failed_long_unchanged; no solver or selection changes.',
        analysis_issue_count=1,solver_errors=0,
        role_counts=dict(Counter(canonical_role(j['case']['role']) for j in m['jobs'])))
    for method in METHODS:
        corrected['summaries'][method]['failed_tail_recovered']=sum(
            r['branches'][method]['recovered'] for r in saved
            if canonical_role(r['job']['case']['role'])=='failed_tail')
    corrected['decision']='bounded_development_signal_only' if corrected['gate']['passed'] else 'no_go'
    require(not corrected['gate']['passed'],'stop review requires a failed mechanism gate')
    with ProcessPoolExecutor(max_workers=20) as pool: data=list(pool.map(audit_job,m['jobs']))
    rows=[r for d in data for r in d['rows']]; methods={}
    for method in METHODS:
        rs=[r for r in rows if r['method']==method and r['applicable']]
        methods[method]=dict(classifications=dict(Counter(r['classification'] for r in rs)),
            altered_attempts=sum(not r['attempted_unchanged_from_old_seed'] for r in rs),
            attempts_different_from_comparator=sum(not r['attempt_same_as_comparator'] for r in rs),
            added_chosen_pair_conditions=sum(bool(r['added_chosen_pairs']) for r in rs),
            removed_chosen_pair_conditions=sum(bool(r['removed_chosen_pairs']) for r in rs),
            added_other_pair_conditions=sum(bool(r['added_other_pairs']) for r in rs),
            total_expanded_including_baseline=sum(d['base_expanded'] for d in data)+sum(r['expanded'] for r in rs))
    coverage=witness_feedback_coverage()
    native_checks=verify_mediation_native()
    summary=dict(schema='lns2.reservation_failure_review.v2',conditions=48,
        verified_new_branches=sum(r['applicable'] for r in rows),native_mediation_paths_verified=native_checks,
        methods=methods,witness_feedback_coverage=coverage,
        witness_present_count=sum(r['present'] for r in coverage),
        recovered_rows=[r for r in rows if r['recovered']],
        base_expanded=sum(d['base_expanded'] for d in data),
        original_result_report_sha256=sha256_file(OUT/'report.json'),
        analysis_issue_count=1,role_counts=corrected['role_counts'],
        decision='stop_automatic_feedback_branch_no_ttf_no_training',
        boundary='Viewed development cases; omission-sensitive avoiders need not be observed collision partners.')
    write_json(OUT/'report-corrected.json',corrected)
    summary['corrected_report_sha256']=sha256_file(OUT/'report-corrected.json')
    write_json(OUT/'round4-rows-v2.json',rows); write_json(OUT/'round4-review-v2.json',summary)
    print(__import__('json').dumps(summary,ensure_ascii=False))


if __name__=='__main__': main()
