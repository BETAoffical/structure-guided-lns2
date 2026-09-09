"""Same-budget prefix feedback, isolated from frozen controllers and old studies."""
from __future__ import annotations

from collections import Counter
import time

from experiments.full_neighborhood_recovery import (
    SearchBudget, check_paths, choose_feedback, needs_recovery, recover, run_sequence,
    scientific_signature,
)
from experiments.local_path_search import constraint_for, pair_events, validate_state
from experiments.native_path_compatibility import paths_of

METHODS = ('ordinary', 'random_retry', 'directed_resources', 'prefix_resources')


def rank_relations(base, order):
    positions = {aid:i for i,aid in enumerate(order)}
    first_times, external = {}, set()
    for record in base['records']:
        victim = record['agent']
        for event in record['incident_events']:
            blocker = event['right'] if event['left']==victim else event['left']
            if blocker not in positions:
                external.add(blocker)
                continue
            if positions[blocker]>=positions[victim]:
                raise ValueError('prefix event contains an unplanned selected agent')
            relation = (blocker,victim)
            first_times[relation] = min(first_times.get(relation,event['time']),event['time'])
    degree = Counter(a for relation in first_times for a in relation)
    rows = [dict(blocker=b,victim=v,first_time=t,blocker_pair_degree=degree[b])
            for (b,v),t in first_times.items()]
    rows.sort(key=lambda r:(-r['blocker_pair_degree'],r['first_time'],
                           positions[r['blocker']],positions[r['victim']],r['blocker'],r['victim']))
    return rows,sorted(external)


def generate_prefix_feedback(probe, state, order, base, seed, budget, limit):
    ranked, external = rank_relations(base,order)
    positions = {aid:i for i,aid in enumerate(order)}
    original = dict(enumerate(paths_of(state)))
    evidence, candidates = [],[]
    for relation in ranked[:limit]:
        blocker,victim = relation['blocker'],relation['victim']
        prefix = base['records'][:positions[victim]]
        overrides = {r['agent']:r['search']['path'] for r in prefix}
        fixed = [i for i in original if i not in positions]+[r['agent'] for r in prefix if r['agent']!=blocker]
        probe.seed_rng(seed)
        relaxed = budget.plan(probe,victim,fixed,overrides)
        evidence.append(dict(relation,relaxed=relaxed))
        if relaxed['status']!='path':
            continue
        blocker_path = overrides[blocker]
        for event in pair_events(blocker_path,relaxed['path']):
            candidates.append(dict(blocker=blocker,victim=victim,
                constraint=list(constraint_for(event,blocker_path)),
                prefix_length=positions[blocker],cap=len(blocker_path)-1))
    return dict(ranked_relations=ranked,evidence=evidence,candidates=candidates,external_blockers=external,
                reason='prefix_internal_feedback' if candidates else 'no_usable_prefix_constraint')


def recover_prefix(probe, state, order, seed, method, config):
    if method not in METHODS:
        raise ValueError('unknown prefix comparison method')
    if method!='prefix_resources':
        return recover(probe,state,order,seed,method,config)
    validate_state(state)
    budget = SearchBudget(config['job_seconds'],config['call_seconds'])
    base = run_sequence(probe,state,order,seed,budget)
    result = dict(status='ok',method=method,base=base,attempts=[],feedback=None,
                  recovered=False,final=base,triggered=needs_recovery(base))
    if result['triggered']:
        feedback = generate_prefix_feedback(probe,state,order,base,seed,budget,config['max_feedback_blockers'])
        result['feedback'] = feedback
        options = choose_feedback(feedback['candidates'],'directed_resources',config['max_attempts'])
        for index,option in enumerate(options):
            if budget.remaining()<=0:
                break
            branch = run_sequence(probe,state,order,seed+index+1,budget,
                prefix=base['records'][:option['prefix_length']],constraint=option['constraint'],
                capped_agent=option['blocker'],cap=option['cap'])
            result['attempts'].append(dict(option=option,seed=seed+index+1,result=branch))
            if branch['status']=='accepted' and branch['strict_decrease']:
                if len(branch['paths'][option['blocker']])-1>option['cap']:
                    raise ValueError('prefix recovery exceeded predecessor cost cap')
                result.update(recovered=True,final=branch)
                break
    final = result['final']
    if final['status'] in ('accepted','rolled_back'):
        result['path_check'] = check_paths(state,final['paths'],order)
    else:
        result['status'] = 'unknown' if final['status']=='unknown' else 'error'
    result['budget_exhausted'] = budget.remaining()<=0 or any(c['status']=='unknown' for c in budget.calls)
    result['expanded'] = sum(c['expanded'] for c in budget.calls)
    result['generated'] = sum(c['generated'] for c in budget.calls)
    result['low_level_calls'] = len(budget.calls)
    result['diagnostic_seconds'] = time.monotonic()-budget.started
    return result


def result_signature(row):
    return dict(base=scientific_signature(row['base']),final=scientific_signature(row['final']),
        attempts=[dict(option=a['option'],seed=a['seed'],result=scientific_signature(a['result'])) for a in row['attempts']],
        recovered=row['recovered'],triggered=row['triggered'],expanded=row['expanded'],generated=row['generated'])


def evaluate_gate(rows, cases, seeds):
    expected = {(c['case_id'],s,m) for c in cases for s in seeds for m in METHODS}
    keys = [(r['job']['case_id'],r['job']['seed'],r['job']['method']) for r in rows]
    if len(keys)!=len(expected) or set(keys)!=expected or any(r['status']!='ok' or r.get('budget_exhausted') for r in rows):
        return dict(passed=False,reason='incomplete_invalid_or_censored_conditions')
    indexed = {key:r for key,r in zip(keys,rows)}
    if any(not r.get('baseline_matches_frozen') or not r.get('native_paths_verified') or
           not r.get('control_matches_frozen',True) for r in rows):
        return dict(passed=False,reason='frozen_baseline_control_or_path_mismatch')
    triggered = {c['case_id']:{s for s in seeds if indexed[(c['case_id'],s,'ordinary')]['triggered']} for c in cases}
    if any(r['triggered']!=(r['job']['seed'] in triggered[r['job']['case_id']]) for r in rows):
        return dict(passed=False,reason='trigger_mismatch')
    stable = {m:set() for m in METHODS}
    for c in cases:
        cid = c['case_id']
        if triggered[cid]!=set(seeds):
            continue
        for method in METHODS:
            if all(indexed[(cid,s,method)]['recovered'] for s in seeds):
                stable[method].add(cid)
    case_by_id = {c['case_id']:c for c in cases}
    main = stable['prefix_resources']
    controls = stable['random_retry']|stable['directed_resources']
    no_harm = all(r['final']['conflicts']<=r['base']['conflicts'] for r in rows)
    checks = dict(at_least_three_double_trigger_states=len(main)>=3,
        at_least_two_maps=len({case_by_id[c]['map_id'] for c in main})>=2,
        failed_tail_recovered=any(case_by_id[c]['role'] in ('failed_long','failed_long_unchanged') for c in main),
        not_fewer_than_either_control=len(main)>=max(len(stable['random_retry']),len(stable['directed_resources'])),
        two_states_exclusive_to_prefix=len(main-controls)>=2,no_harm_to_ordinary=no_harm)
    return dict(passed=all(checks.values()),checks=checks,
        stable_recovered_states={m:sorted(v) for m,v in stable.items()},
        trigger_strata=dict(Counter(str(len(v)) for v in triggered.values())),
        interpretation='development_mechanism_only_requires_future_independent_validation')
