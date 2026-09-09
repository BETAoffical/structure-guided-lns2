"""Reconstruct saved PP conflict budgets without planning or selecting actions."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

from experiments._common import read_json, write_json, write_jsonl
from experiments.full_recovery_collection import check_output, load, output_file
from experiments.local_path_compatibility import ROOT, contained, run_lock, sha256_file
from experiments.local_path_search import validate_state
from experiments.native_path_compatibility import check_seal, seal
from experiments.state_analysis import reconstruct_conflicts

SCHEMA = 'lns2.conflict_budget_attribution.v1'
SOURCE = 'build/initlns-full-neighborhood-recovery-v1-final'
OUTPUT = 'build/initlns-conflict-budget-attribution-v1'
EVIDENCE = 'artifacts/initlns-full-neighborhood-recovery-v1/evidence.json'


def events_of(agents):
    return [dict(left=e.left, right=e.right, time=e.time, kind=e.kind)
            for e in reconstruct_conflicts(agents)]


def event_key(event):
    return (event['left'], event['right'], event['time'], event['kind'])


def pair_set(events):
    return {tuple(sorted((e['left'], e['right']))) for e in events}


def optimistic_cover(pairs, limit=2):
    """Incident-edge coverage only: paths may not realize this deletion bound."""
    pairs = set(map(tuple, pairs))
    agents = sorted({a for edge in pairs for a in edge})
    choices = [()] + [ids for n in range(1, min(limit, len(agents))+1)
                      for ids in combinations(agents, n)]
    best = min(choices, key=lambda ids: (-sum(bool(set(ids)&set(e)) for e in pairs), len(ids), ids))
    return dict(agents=list(best), covered_pairs=sum(bool(set(best)&set(e)) for e in pairs))


def reconstruct_ledger(state, order, sequence):
    graph, agents = validate_state(state)
    selected = set(order)
    if not order or len(selected)!=len(order) or not selected<=set(agents):
        raise ValueError('invalid neighborhood order')
    original_pairs = pair_set(events_of(state['agents']))
    allowance = sum(bool(selected&set(e)) for e in original_pairs)
    records = sequence['records']
    if len(records)>len(order) or [r['agent'] for r in records]!=order[:len(records)]:
        raise ValueError('record order mismatch')
    visible = {aid:a['path'] for aid,a in agents.items() if aid not in selected}
    cumulative, steps = set(), []
    for index, record in enumerate(records):
        aid, search = record['agent'], record['search']
        if search['status']!='path':
            if index!=len(records)-1 or sequence['status'] not in ('not_found','unknown'):
                raise ValueError('unexpected incomplete search')
            break
        path = search['path']
        if not path or path[0]!=agents[aid]['start'] or path[-1]!=agents[aid]['goal']:
            raise ValueError('attempt path endpoints')
        if any(p not in graph for p in path) or any(v not in graph[u] for u,v in zip(path,path[1:])):
            raise ValueError('attempt path geometry')
        if search['cost']!=len(path)-1:
            raise ValueError('attempt path cost')
        visible[aid] = path
        events = [e for e in events_of([dict(id=i,path=p) for i,p in visible.items()])
                  if aid in (e['left'],e['right'])]
        if Counter(map(event_key,events))!=Counter(map(event_key,record['incident_events'])):
            raise ValueError('saved incident events differ from paths')
        incident = pair_set(events)
        added = incident-cumulative
        before = set(cumulative)
        cumulative |= incident
        if record['cumulative_pairs']!=len(cumulative):
            raise ValueError('cumulative unique-pair count mismatch')
        internal = {e for e in cumulative if set(e)<=selected}
        steps.append(dict(index=index,agent=aid,event_count=len(events),incident_pairs=sorted(incident),
            added_pairs=sorted(added),cumulative_pairs=sorted(cumulative),internal_pairs=sorted(internal),
            external_pairs=sorted(cumulative-internal),budget_before=allowance-len(before),
            budget_after=allowance-len(cumulative)))
        if len(cumulative)>allowance and (sequence['status']!='rolled_back' or index!=len(records)-1):
            raise ValueError('official early rollback boundary mismatch')
    status = sequence['status']
    if status in ('rolled_back','accepted'):
        if sequence['old_pairs']!=allowance or sequence['attempted_pairs']!=len(cumulative):
            raise ValueError('official incident allowance mismatch')
        original_paths = [a['path'] for a in state['agents']]
        if status=='rolled_back':
            if len(cumulative)<=allowance or sequence['paths']!=original_paths:
                raise ValueError('invalid rollback')
        else:
            final = [visible[a['id']] for a in state['agents']]
            if len(records)!=len(order) or sequence['paths']!=final:
                raise ValueError('incomplete accepted sequence')
            total = len(pair_set(events_of([dict(id=a['id'],path=p) for a,p in zip(state['agents'],final)])))
            if total!=sequence['conflicts'] or sequence['same_paths']!=(final==original_paths):
                raise ValueError('accepted result mismatch')
    elif status not in ('not_found','unknown'):
        raise ValueError('unknown sequence status')
    internal = {e for e in cumulative if set(e)<=selected}
    earlier = set(map(tuple,steps[-2]['internal_pairs'])) if len(steps)>1 else set()
    last_internal = {e for e in pair_set(records[-1]['incident_events']) if set(e)<=selected} if records else set()
    excess = max(0,len(cumulative)-allowance)
    return dict(status=status,allowance=allowance,initial_global_pairs=len(original_pairs),
        steps=steps,attempted_pairs=len(cumulative),excess=excess,
        internal_pairs=sorted(internal),external_pairs=sorted(cumulative-internal),
        earlier_internal_pairs=sorted(earlier),last_internal_pairs=sorted(last_internal),
        earlier_internal_deletion_could_cover_excess=status=='rolled_back' and len(earlier)>=excess,
        any_internal_deletion_could_cover_excess=status=='rolled_back' and len(internal)>=excess,
        best_two_internal_cover=optimistic_cover(internal),
        interpretation='optimistic_pair_deletion_bound_not_path_feasibility')


def prefix_difference(base, alternative):
    length = min(len(base['steps']),len(alternative['steps']))
    if not length:
        return dict(common_processed_agents=0)
    a,b = base['steps'][length-1],alternative['steps'][length-1]
    result = dict(common_processed_agents=length,not_a_full_outcome_comparison=True)
    for kind in ('internal_pairs','external_pairs'):
        old,new = set(map(tuple,a[kind])),set(map(tuple,b[kind]))
        result[kind] = dict(removed=sorted(old-new),added=sorted(new-old))
    return result


def audit_condition(payload):
    for path,expected in payload['files'].items():
        if sha256_file(contained(path))!=expected:
            raise ValueError('worker input SHA changed: '+path)
    state = read_json(contained(payload['case']['state_file']))
    transition = read_json(contained(payload['case']['historical_transition_file']))
    order = transition['metrics']['repair_order']
    row = read_json(contained(payload['row_file']))
    check_seal(row)
    base = reconstruct_ledger(state,order,row['base'])
    attempts = []
    for index,attempt in enumerate(row['attempts']):
        ledger = reconstruct_ledger(state,order,attempt['result'])
        attempts.append(dict(index=index,option=attempt['option'],ledger=ledger,
                             common_prefix_change=prefix_difference(base,ledger)))
    return seal(dict(schema=SCHEMA,fingerprint=payload['fingerprint'],job=row['job'],
        role=payload['case']['role'],map_id=payload['case']['map_id'],
        triggered=row['triggered'],recovered=row['recovered'],base=base,attempts=attempts,
        previous_external_feedback_only=bool(row['feedback'] and not row['feedback']['candidates']
                                            and row['feedback']['external_blockers'])))


def summarize(rows):
    failed = [r for r in rows if r['base']['status']=='rolled_back']
    hidden = [r for r in failed if r['previous_external_feedback_only'] and r['base']['earlier_internal_pairs']]
    attempts = [a for r in rows for a in r['attempts']]
    changed = [a for a in attempts if a['common_prefix_change'].get('common_processed_agents',0)>0]
    return dict(schema=SCHEMA,conditions=len(rows),states=len({r['job']['case_id'] for r in rows}),
        maps=len({r['map_id'] for r in rows}),rolled_back_conditions=len(failed),
        statuses=dict(Counter(r['base']['status'] for r in rows)),
        missing_earlier_internal_feedback=len(hidden),
        missing_earlier_internal_feedback_keys=[r['job'] for r in hidden],
        earlier_internal_bound_covers_excess=sum(r['base']['earlier_internal_deletion_could_cover_excess'] for r in failed),
        any_internal_bound_covers_excess=sum(r['base']['any_internal_deletion_could_cover_excess'] for r in failed),
        no_recorded_internal_pairs=sum(not r['base']['internal_pairs'] for r in failed),
        best_two_internal_bound_covers_excess=sum(r['base']['best_two_internal_cover']['covered_pairs']>=r['base']['excess'] for r in failed),
        attempt_count=len(attempts),attempt_statuses=dict(Counter(a['ledger']['status'] for a in attempts)),
        attempts_remove_internal_but_add_external=sum(bool(a['common_prefix_change']['internal_pairs']['removed']) and
            bool(a['common_prefix_change']['external_pairs']['added']) for a in changed),
        attempts_reach_different_prefix_length=sum(len(a['ledger']['steps'])!=len(r['base']['steps']) for r in rows for a in r['attempts']),
        decision='attribution_only_no_recovery_or_timing_admission',solver_calls=0,timing_allowed=False)


def run(output, workers=20, resume=False):
    if not 1<=workers<=20:
        raise ValueError('workers must be within 1..20')
    if not output.resolve().is_relative_to((ROOT/'build').resolve()) or output.resolve()==ROOT/'build':
        raise ValueError('output must be strictly inside build')
    source = contained(SOURCE)
    if output.resolve().is_relative_to(source) or source.is_relative_to(output.resolve()):
        raise ValueError('audit output overlaps protected source results')
    evidence = read_json(contained(EVIDENCE))
    m = load(source)
    for name,expected in evidence['files'].items():
        if sha256_file(source/name)!=expected:
            raise ValueError('frozen source evidence changed: '+name)
    report = read_json(source/'report.json')
    check_seal(report)
    if not report['complete'] or not report['parity_passed'] or report['fingerprint']!=m['content_sha256']:
        raise ValueError('source is not complete authenticated evidence')
    source_hashes = {}
    for job in m['jobs']:
        path = output_file(source,job)
        relative = path.relative_to(ROOT).as_posix()
        expected = report['result_hashes'][relative]
        if sha256_file(path)!=expected:
            raise ValueError('source result changed')
        row = read_json(path)
        check_output(m,job,row)
        source_hashes[relative] = expected
    impl = ['experiments/conflict_budget_attribution.py','scripts/audit_conflict_budget_attribution.py']
    registration = seal(dict(schema=SCHEMA,source_manifest=m['content_sha256'],evidence_sha=sha256_file(contained(EVIDENCE)),
        implementation={p:sha256_file(contained(p)) for p in impl},source_results=source_hashes,
        policy='saved-path attribution only; no solver, runtime mutation or recovery promotion'))
    cases = {c['case_id']:c for c in m['cases']}
    with run_lock(output):
        manifest_path = output/'manifest.json'
        if manifest_path.exists():
            if not resume or read_json(manifest_path)!=registration:
                raise ValueError('existing output requires identical registration and resume')
        else:
            write_json(manifest_path,registration)
        jobs = [j for j in m['jobs'] if j['method']=='directed_resources']
        rows,pending = [],[]
        for job in jobs:
            destination = output/'conditions'/(job['job_id']+'.json')
            if destination.exists():
                saved = read_json(destination)
                check_seal(saved)
                if saved['job']!=job or saved['fingerprint']!=registration['content_sha256']:
                    raise ValueError('saved condition identity mismatch')
                rows.append(saved)
                continue
            case = cases[job['case_id']]
            row_file = output_file(source,job).relative_to(ROOT).as_posix()
            files = {p:m['files'][p] for p in (case['state_file'],case['historical_transition_file'])}
            files[row_file] = source_hashes[row_file]
            pending.append(dict(case=case,row_file=row_file,files=files,fingerprint=registration['content_sha256']))
        write_json(output/'run_status.json',dict(status='running',workers=workers,completed=len(rows),total=len(jobs)))
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(audit_condition,p) for p in pending]
                for future in as_completed(futures):
                    row = future.result()
                    write_json(output/'conditions'/(row['job']['job_id']+'.json'),row)
                    rows.append(row)
                    write_json(output/'run_status.json',dict(status='running',workers=workers,completed=len(rows),total=len(jobs)))
                    print(f"verified {len(rows)}/{len(jobs)} {row['job']['case_id']} seed={row['job']['seed']}",flush=True)
            rows.sort(key=lambda r:(r['job']['case_id'],r['job']['seed']))
            result = seal(dict(summarize(rows),fingerprint=registration['content_sha256'],
                condition_hashes={r['job']['job_id']:r['content_sha256'] for r in rows}))
            write_jsonl(output/'conditions.jsonl',rows)
            write_json(output/'report.json',result)
            write_json(output/'run_status.json',dict(status='completed',workers=workers,completed=len(rows),total=len(jobs)))
            return result
        except BaseException as error:
            write_json(output/'run_status.json',dict(status='failed',completed=len(rows),total=len(jobs),error=str(error)))
            raise
