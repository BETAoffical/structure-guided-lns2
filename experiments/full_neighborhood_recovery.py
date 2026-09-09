"""Bounded soft-PP recovery on the original neighborhood, isolated from runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
import time

from experiments.local_path_search import constraint_for, pair_events, validate_state
from experiments.native_path_compatibility import edge_set, paths_of
from experiments.state_analysis import reconstruct_conflicts

METHODS = ('ordinary', 'random_retry', 'directed_events', 'directed_resources')


@dataclass
class SearchBudget:
    seconds: float
    call_seconds: float
    started: float = field(default_factory=time.monotonic)
    calls: list = field(default_factory=list)

    def remaining(self):
        return max(0., self.seconds-(time.monotonic()-self.started))

    def plan(self, probe, aid, fixed, overrides, constraints=(), cap=-1):
        found = probe.plan(aid, fixed, overrides, False, list(constraints), cap,
                           min(self.call_seconds, self.remaining()))
        self.calls.append(dict(agent=aid, **found))
        return found


def validate_order(state, order):
    ids = [a['id'] for a in state['agents']]
    if ids != list(range(len(ids))):
        raise ValueError('native scenario IDs must be ordered and contiguous')
    if not order or len(set(order)) != len(order) or not set(order) <= set(ids):
        raise ValueError('invalid complete neighborhood order')


def run_sequence(probe, state, order, seed, budget, prefix=(), constraint=None, capped_agent=None, cap=-1):
    """Each branch starts at the same state; only validated prefix paths are reused."""
    validate_order(state, order)
    if len(prefix) > len(order) or [r['agent'] for r in prefix] != order[:len(prefix)]:
        raise ValueError('prefix is not the original ordered prefix')
    original = dict(enumerate(paths_of(state)))
    visible = {i: p for i, p in original.items() if i not in order}
    fixed = list(visible)
    overrides, records, pairs = {}, [], set()
    old_pairs = edge_set(original, order)
    probe.seed_rng(seed)
    for index, aid in enumerate(order):
        if index < len(prefix):
            found = prefix[index]['search']
        else:
            if budget.remaining() <= 0:
                return dict(status='unknown', reason='job_budget', records=records)
            found = budget.plan(probe, aid, fixed, overrides,
                [constraint] if aid == capped_agent and constraint is not None else (),
                cap if aid == capped_agent else -1)
        if found['status'] != 'path':
            records.append(dict(agent=aid, search=found, incident_events=[]))
            return dict(status='unknown' if found['status'] == 'unknown' else 'not_found',
                        reason='single_agent_' + found['status'], records=records)
        overrides[aid] = found['path']
        visible[aid] = found['path']
        events = [e for e in reconstruct_conflicts([dict(id=i, path=p) for i, p in visible.items()])
                  if aid in (e.left, e.right)]
        pairs.update((e.left, e.right) for e in events)
        records.append(dict(agent=aid, search=found, cumulative_pairs=len(pairs),
            incident_events=[dict(left=e.left, right=e.right, time=e.time, kind=e.kind) for e in events]))
        if len(pairs) > len(old_pairs):
            return dict(status='rolled_back', reason='conflict_bound_exceeded', records=records,
                attempted_pairs=len(pairs), old_pairs=len(old_pairs), paths=paths_of(state),
                conflicts=state['num_of_colliding_pairs'], strict_decrease=False, same_paths=True)
        fixed.append(aid)
    final = [overrides.get(i, p) for i, p in original.items()]
    remaining = len(edge_set(dict(enumerate(final))))
    return dict(status='accepted', reason='none', records=records, attempted_pairs=len(pairs),
        old_pairs=len(old_pairs), paths=final, conflicts=remaining,
        strict_decrease=remaining < state['num_of_colliding_pairs'], same_paths=final == paths_of(state))


def needs_recovery(result):
    return result['status'] == 'rolled_back' or (result['status'] == 'accepted' and result['same_paths'])


def choose_feedback(candidates, method, limit):
    ordered = sorted(candidates, key=lambda c: (c['constraint'][1], c['blocker'], c['victim'], tuple(c['constraint'])))
    chosen, seen = [], set()
    for row in ordered:
        kind, tick, u, v = row['constraint']
        key = (row['blocker'], kind, u, v) if method == 'directed_resources' else (row['blocker'], kind, tick, u, v)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(row)
        if len(chosen) == limit:
            break
    return chosen


def generate_feedback(probe, state, order, base, seed, budget, max_blockers):
    positions = {aid: i for i, aid in enumerate(order)}
    original = dict(enumerate(paths_of(state)))
    records = base['records']
    candidates, evidence, external = [], [], set()
    # Bound failures use the failed insertion. Unchanged accepted paths use the first conflicting insertion.
    targets = records[-1:] if base['status'] == 'rolled_back' else [r for r in records if r['incident_events']][:1]
    resources = []
    for record in targets:
        victim = record['agent']
        for event in record['incident_events']:
            blocker = event['right'] if event['left'] == victim else event['left']
            if blocker not in positions:
                external.add(blocker)
            elif positions[blocker] < positions[victim]:
                resources.append((event['time'], blocker, victim))
    for _, blocker, victim in sorted(set(resources)):
        if len(evidence) >= max_blockers:
            break
        if any(e['blocker'] == blocker and e['victim'] == victim for e in evidence):
            continue
        prefix = records[:positions[victim]]
        overrides = {r['agent']: r['search']['path'] for r in prefix}
        fixed = [i for i in original if i not in positions] + [r['agent'] for r in prefix if r['agent'] != blocker]
        probe.seed_rng(seed)
        relaxed = budget.plan(probe, victim, fixed, overrides)
        record = dict(blocker=blocker, victim=victim, relaxed=relaxed)
        evidence.append(record)
        if relaxed['status'] != 'path':
            continue
        blocker_path = overrides[blocker]
        for event in pair_events(blocker_path, relaxed['path']):
            candidates.append(dict(blocker=blocker, victim=victim,
                constraint=list(constraint_for(event, blocker_path)),
                prefix_length=positions[blocker], cap=len(blocker_path)-1))
    return dict(candidates=candidates, evidence=evidence, external_blockers=sorted(external),
                reason='internal_feedback' if candidates else 'no_internal_feedback')


def check_paths(state, paths, order):
    original = paths_of(state)
    if len(paths) != len(original):
        raise ValueError('wrong path count')
    if any(paths[i] != original[i] for i in range(len(paths)) if i not in order):
        raise ValueError('external path modified')
    agents = [dict(a, path=p) for a, p in zip(state['agents'], paths)]
    edges = sorted(edge_set(dict(enumerate(paths))))
    validate_state(dict(state, agents=agents, conflict_edges=edges, num_of_colliding_pairs=len(edges)))
    if len(edges) > state['num_of_colliding_pairs']:
        raise ValueError('accepted result increases total conflicts')
    return dict(conflicts=len(edges), globally_feasible=not edges,
        changed_agents=[i for i in order if paths[i] != original[i]],
        sum_of_costs=sum(len(p)-1 for p in paths), makespan=max(len(p)-1 for p in paths))


def recover(probe, state, order, seed, method, config):
    if method not in METHODS:
        raise ValueError('unknown recovery method')
    validate_state(state)
    budget = SearchBudget(config['job_seconds'], config['call_seconds'])
    base = run_sequence(probe, state, order, seed, budget)
    result = dict(status='ok', method=method, base=base, attempts=[], feedback=None,
                  recovered=False, final=base, triggered=needs_recovery(base))
    if method != 'ordinary' and result['triggered']:
        if method == 'random_retry':
            options = [None] * config['max_attempts']
        else:
            feedback = generate_feedback(probe, state, order, base, seed, budget, config['max_feedback_blockers'])
            result['feedback'] = feedback
            options = choose_feedback(feedback['candidates'], method, config['max_attempts'])
        for index, option in enumerate(options):
            if budget.remaining() <= 0:
                break
            branch = run_sequence(probe, state, order, seed+index+1, budget,
                prefix=base['records'][:option['prefix_length']] if option else (),
                constraint=option['constraint'] if option else None,
                capped_agent=option['blocker'] if option else None, cap=option['cap'] if option else -1)
            result['attempts'].append(dict(option=option, seed=seed+index+1, result=branch))
            if branch['status'] == 'accepted' and branch['strict_decrease']:
                if option and len(branch['paths'][option['blocker']])-1 > option['cap']:
                    raise ValueError('reselected predecessor exceeds baseline cost')
                result.update(recovered=True, final=branch)
                break
    final = result['final']
    if final['status'] in ('accepted', 'rolled_back'):
        result['path_check'] = check_paths(state, final['paths'], order)
    else:
        result['status'] = 'unknown' if final['status'] == 'unknown' else 'error'
    result['budget_exhausted'] = budget.remaining() <= 0 or any(c['status'] == 'unknown' for c in budget.calls)
    result['expanded'] = sum(c['expanded'] for c in budget.calls)
    result['generated'] = sum(c['generated'] for c in budget.calls)
    result['low_level_calls'] = len(budget.calls)
    result['diagnostic_seconds'] = time.monotonic()-budget.started
    return result


def scientific_signature(sequence):
    return {k: sequence[k] for k in ('status', 'paths', 'attempted_pairs', 'same_paths') if k in sequence} | {
        'records': [dict(agent=r['agent'], **{k: r['search'][k]
            for k in ('status', 'path', 'cost', 'expanded', 'generated', 'low_level_collisions')}) for r in sequence['records']]}


def mechanism_gate(rows, cases, expected_count):
    if len(rows) != expected_count or any(r['status'] != 'ok' or r.get('budget_exhausted') for r in rows):
        return dict(passed=False, reason='incomplete_error_or_censored_evidence')
    keys = {(r['job']['case_id'], r['job']['seed'], r['job']['method']) for r in rows}
    seeds = {r['job']['seed'] for r in rows}
    expected_keys = {(c['case_id'], seed, method) for c in cases for seed in seeds for method in METHODS}
    if len(keys) != len(rows) or keys != expected_keys:
        return dict(passed=False, reason='duplicate_or_missing_paired_condition')
    source = {c['case_id']: c for c in cases}
    success = {method: {} for method in METHODS}
    for row in rows:
        job = row['job']
        if row['recovered']:
            success[job['method']].setdefault(job['case_id'], set()).add(job['seed'])
    stable = {method: {c for c, seen in found.items() if seen == seeds} for method, found in success.items()}
    main = stable['directed_resources']
    checks = dict(at_least_three_states=len(main) >= 3,
        at_least_two_maps=len({source[c]['map_id'] for c in main}) >= 2,
        failed_tail_recovery=any(source[c]['role'] in ('failed_long_unchanged', 'failed_long') for c in main),
        not_fewer_than_random=len(main) >= len(stable['random_retry']),
        at_least_two_exclusive_states=len(main-stable['random_retry']) >= 2)
    return dict(passed=all(checks.values()), checks=checks, stable_recoveries={k: sorted(v) for k, v in stable.items()},
                interpretation='mechanism_admission_only_not_runtime_promotion')
