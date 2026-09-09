"""Bounded paired-path lookahead; never installed in the official controller."""
from __future__ import annotations

import time

from experiments.full_neighborhood_recovery import SearchBudget, check_paths, needs_recovery, run_sequence
from experiments.local_path_compatibility import digest
from experiments.local_path_search import at, pair_events, validate_state
from experiments.prefix_budget_feedback import generate_prefix_feedback

METHODS = ('whole_pair', 'whole_pair_guarded')


def avoid_path(path, cap):
    """Complete hard occupancy under a finite cost cap, including both goal waits."""
    if not path or cap < 0:
        raise ValueError('nonempty path and finite nonnegative cap required')
    horizon = max(cap, len(path)-1)
    constraints = {('vertex', t, at(path,t), at(path,t)) for t in range(horizon+1)}
    constraints.update(('edge', t, path[t], path[t-1]) for t in range(1,len(path))
                       if path[t]!=path[t-1])
    return constraints


def external_pairs(path, state, order):
    return {a['id'] for a in state['agents'] if a['id'] not in order and pair_events(path,a['path'])}


def options_of(feedback, order, base, limit):
    positions = {aid:i for i,aid in enumerate(order)}
    usable = [r for r in feedback['evidence'] if r['relaxed']['status']=='path']
    # Two repetitions per relation, interleaved before repeating a relation.
    return [dict(blocker=r['blocker'],victim=r['victim'],reference=r['relaxed'],repeat=trial,
                 prefix_length=positions[r['blocker']],
                 cap=len(base['records'][positions[r['blocker']]]['search']['path'])-1)
            for trial in range(2) for r in usable][:limit]


class PairProbe:
    """Constrain the predecessor and reuse the already charged lookahead path."""
    def __init__(self, probe, option, constraints):
        self.probe, self.option, self.constraints = probe, option, constraints

    def seed_rng(self, seed):
        self.probe.seed_rng(seed)

    def plan(self, aid, fixed, overrides, hard, constraints, cap, seconds):
        if hard or constraints:
            raise ValueError('unexpected mixed diagnostic constraints')
        if aid==self.option['victim']:
            return dict(self.option['reference'],expanded=0,generated=0,search_seconds=0.,
                        wrapper_seconds=0.,cached_reference=True,low_level_collisions=None,
                        reference_low_level_collisions=self.option['reference']['low_level_collisions'])
        if aid==self.option['blocker']:
            return self.probe.plan(aid,fixed,overrides,False,self.constraints,self.option['cap'],seconds)
        return self.probe.plan(aid,fixed,overrides,False,[],cap,seconds)


def branch_constraints(state, order, base, option, guarded):
    blocker = option['blocker']
    base_path = next(r['search']['path'] for r in base['records'] if r['agent']==blocker)
    allowed = external_pairs(base_path,state,order)
    constraints = avoid_path(option['reference']['path'],option['cap'])
    if guarded:
        for agent in state['agents']:
            if agent['id'] not in order and agent['id'] not in allowed:
                constraints |= avoid_path(agent['path'],option['cap'])
    return sorted(constraints), sorted(allowed)


def verify_pair_branch(state, order, base, attempt, guarded):
    option, branch = attempt['option'],attempt['result']
    if branch is None:
        if not guarded or attempt['reason']!='reference_adds_external_pair':
            raise ValueError('unexplained unexecuted option')
        original = next(r['search']['path'] for r in base['records'] if r['agent']==option['victim'])
        if not external_pairs(option['reference']['path'],state,order)-external_pairs(original,state,order):
            raise ValueError('unsupported reference guard rejection')
        return dict(classification='reference_guard_rejected')
    found = {r['agent']:r['search'] for r in branch['records']}
    blocker, victim = option['blocker'],option['victim']
    path = found.get(blocker,{})
    constraints, allowed = branch_constraints(state,order,base,option,guarded)
    if attempt['constraint_sha256']!=digest(constraints) or attempt['constraint_count']!=len(constraints):
        raise ValueError('hard occupancy source mismatch')
    if branch['records'][:option['prefix_length']]!=base['records'][:option['prefix_length']]:
        raise ValueError('preserved PP prefix changed')
    if path.get('status')!='path':
        return dict(classification='blocker_path_unavailable')
    if len(path['path'])-1>option['cap'] or pair_events(path['path'],option['reference']['path']):
        raise ValueError('whole-pair cost or occupancy violation')
    for kind,t,u,v in constraints:
        violated = at(path['path'],t)==u if kind=='vertex' else at(path['path'],t-1)==u and at(path['path'],t)==v
        if violated:
            raise ValueError('hard constraint violated')
    if guarded and external_pairs(path['path'],state,order)-set(allowed):
        raise ValueError('predecessor external guard violated')
    other = found.get(victim,{})
    if other.get('status')!='path':
        return dict(classification='victim_not_inserted')
    if other['path']!=option['reference']['path'] or not other.get('cached_reference'):
        raise ValueError('committed lookahead path changed')
    if pair_events(path['path'],other['path']):
        raise ValueError('committed pair collides')
    if guarded:
        original = next(r['search']['path'] for r in base['records'] if r['agent']==victim)
        if external_pairs(other['path'],state,order)-external_pairs(original,state,order):
            raise ValueError('victim external guard violated')
    return dict(classification='pair_removed',branch_status=branch['status'])


def recover_pair(probe, state, order, seed, method, config):
    if method not in METHODS:
        raise ValueError('unknown whole-pair method')
    validate_state(state)
    budget = SearchBudget(config['job_seconds'],config['call_seconds'])
    base = run_sequence(probe,state,order,seed,budget)
    result = dict(status='ok',method=method,base=base,final=base,triggered=needs_recovery(base),
                  recovered=False,feedback=None,attempts=[])
    guarded = method=='whole_pair_guarded'
    if result['triggered']:
        feedback = generate_prefix_feedback(probe,state,order,base,seed,budget,config['max_feedback_blockers'])
        result['feedback'] = feedback
        for index,option in enumerate(options_of(feedback,order,base,config['max_attempts'])):
            if budget.remaining()<=0:
                break
            attempt = dict(option=option,seed=seed+index+1,result=None)
            original = next(r['search']['path'] for r in base['records'] if r['agent']==option['victim'])
            if guarded and external_pairs(option['reference']['path'],state,order)-external_pairs(original,state,order):
                attempt['reason']='reference_adds_external_pair'
            else:
                constraints,_ = branch_constraints(state,order,base,option,guarded)
                adapter = PairProbe(probe,option,constraints)
                attempt.update(constraint_count=len(constraints),constraint_sha256=digest(constraints),
                    result=run_sequence(adapter,state,order,seed+index+1,budget,
                                        prefix=base['records'][:option['prefix_length']]))
            attempt['pair_check'] = verify_pair_branch(state,order,base,attempt,guarded)
            result['attempts'].append(attempt)
            branch = attempt['result']
            if branch and branch['status']=='accepted' and branch['strict_decrease']:
                result.update(recovered=True,final=branch)
                break
    if result['final']['status'] in ('accepted','rolled_back'):
        result['path_check']=check_paths(state,result['final']['paths'],order)
    else:
        result['status']='unknown' if result['final']['status']=='unknown' else 'error'
    result.update(budget_exhausted=budget.remaining()<=0 or any(c['status']=='unknown' for c in budget.calls),
                  expanded=sum(c['expanded'] for c in budget.calls),generated=sum(c['generated'] for c in budget.calls),
                  low_level_calls=sum(not c.get('cached_reference',False) for c in budget.calls),
                  cached_insertions=sum(c.get('cached_reference',False) for c in budget.calls),
                  diagnostic_seconds=time.monotonic()-budget.started)
    return result
